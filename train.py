import torchaudio
from tools.compute_metrics import compute_metrics
from models.generator import TSCNet
from models import discriminator
import os
from data import dataloader
import torch.nn.functional as F
import torch
from utils import power_compress, power_uncompress, multi_res_stft_loss, to_complex
import logging
from torchinfo import summary
import argparse
import soundfile as sf
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=60, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000 * 2, help="cut length, default is 2 seconds in denoise "
                                                                   "and dereverberation")
parser.add_argument("--data_dir", type=str, default='.\Dataset',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='./saved_model',
                    help="dir of saved model")
# Paper argument for loss_weights
parser.add_argument("--loss_weights", type=float, nargs=5, default=[0.1, 0.9, 0.2, 0.05, 0.04],
                    help="weights of RI components, magnitude, time loss, and Metric Disc, MR STFT loss")
# Original argument for loss_weights
# parser.add_argument("--loss_weights", nargs='+',type=float,default=[0.1, 0.9, 0.2, 0.05],
#                     help="weights of RI components, magnitude, time loss, and Metric Disc")
parser.add_argument("--attn1", type=str, default=None,
            help="attention block to use, options: 'se', 'cbam', 'simam', 'eca', or None")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def to_complex(tensor_2channel: torch.Tensor):
    # tensor_2channel should be [B, 2, F, T]
    assert tensor_2channel.size(1) == 2, "Channel dimension must be 2 for real/imag"
    tensor = tensor_2channel.permute(0, 2, 3, 1).contiguous()  # → [B, F, T, 2]
    return torch.view_as_complex(tensor)

def enhance_one_track(
    model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False
):
    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)
    # if sr != 16000:
    #     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    #     noisy = resampler(noisy)
    #     sr = 16000
    assert sr == 16000
    noisy = noisy.cuda()

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len / cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True,return_complex=True
    )
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    # est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_spec_uncompress = power_uncompress(est_real, est_imag)
    est_complex = to_complex(est_spec_uncompress)

    est_audio = torch.istft(
        est_complex,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).cuda(),
        onesided=True,
    )
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length

class Trainer:
    def __init__(self, train_ds, test_ds):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.warmup_epochs = 0
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1,attn1=args.attn1).cuda()
        summary(self.model, [(1, 2, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)])
        self.discriminator = discriminator.Discriminator(ndf=16).cuda()
        summary(self.discriminator, [(1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
                                     (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1)])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=2 * args.init_lr)

    def train_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(args.batch_size).cuda()
        # Normalisation
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))  
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)  # [B,N]
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True, return_complex=True)  # [B,F,T]
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True, return_complex=True)  # [B,F,T]

        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)  # [B, 2, T, F]
        # print(noisy_spec.shape)
        clean_spec = power_compress(clean_spec)  # [B, 2, F, T]
        # print(clean_spec.shape)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)  # [B, 1, F, T]
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)  # [B, 1, F, T]

        est_real, est_imag = self.model(noisy_spec)  # eat_real, est_imag [B, 1, T, F]
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)  # [B, 1, F, T]
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)  # [B,1,F,T]
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)  # [B,1,F,T]

        est_spec_uncompress = power_uncompress(est_real, est_imag)  # # [B,2,F,T]
        # print("est_spec_uncompress shape:", est_spec_uncompress.shape)

        est_complex = to_complex(est_spec_uncompress)  # [B, F, T], complex

        est_audio = torch.istft(est_complex, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(), onesided=True)
        spec_loss = multi_res_stft_loss(est_audio, clean)  

        length = est_audio.size(-1)
        # calculate generator loss
        if self.current_epoch >= self.warmup_epochs:
            self.optimizer.zero_grad()
            predict_fake_metric = self.discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

            loss_mag = F.mse_loss(est_mag, clean_mag)
            loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
            time_loss = torch.mean(torch.abs(est_audio - clean))
            loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
                   + args.loss_weights[3] * gen_loss_GAN + args.loss_weights[4] * spec_loss
            # loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
            #        + args.loss_weights[3] * gen_loss_GAN 
            loss.backward()
            self.optimizer.step()
        else:
            loss = torch.zeros(1).cuda()

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        if pesq_score is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])

        # return loss.item(), discrim_loss_metric.item(),predict_enhance_metric,predict_max_metric
        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(args.batch_size).cuda()

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        # --------------------
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True, return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True, return_complex=True)

        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)  # [B, 2, T, F]
        # print(noisy_spec.shape)
        clean_spec = power_compress(clean_spec)  # [B, 2, F, T]
        # print(clean_spec.shape)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)  # [B, 1, F, T]
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)  # [B, 1, F, T]

        est_real, est_imag = self.model(noisy_spec)  # eat_real, est_imag [B, 1, T, F]
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)  # [B, 1, F, T]
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)  # [B,1,F,T]
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)  # [B,1,F,T]

        est_spec_uncompress = power_uncompress(est_real, est_imag)  # [B,2,F,T]
        # print("est_spec_uncompress shape:", est_spec_uncompress.shape)

        est_complex = to_complex(est_spec_uncompress)  # [B, F, T], complex

        est_audio = torch.istft(est_complex, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(), onesided=True)
        spec_loss = multi_res_stft_loss(est_audio, clean)
        # -----------------
        # calculate loss
        predict_fake_metric = self.discriminator(clean_mag, est_mag)
        gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

        time_loss = torch.mean(torch.abs(est_audio - clean))
        length = est_audio.size(-1)
        loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
               + args.loss_weights[3] * gen_loss_GAN + args.loss_weights[4] * spec_loss
        # loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
        #            + args.loss_weights[3] * gen_loss_GAN 
        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = 'Generator loss: {}, Discriminator loss: {}'
        logging.info(
            template.format(gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    @torch.no_grad()
    def evaluate_epoch(self, noisy_dir, clean_dir):
        self.model.eval()
        audio_list = os.listdir(noisy_dir)
        metrics_total = np.zeros(6)
        num = len(audio_list)

        for audio in audio_list:
            noisy_path = os.path.join(noisy_dir, audio)
            clean_path = os.path.join(clean_dir, audio)
            est_audio, length = enhance_one_track(
                    self.model, noisy_path, None, 16000 * 16,
                    self.n_fft, self.n_fft // 4, False
                )

            clean_audio, sr = sf.read(clean_path)
            assert sr == 16000
            metrics = compute_metrics(clean_audio, est_audio, sr, 0)
            metrics_total += np.array(metrics)

        metrics_avg = metrics_total / num
        logging.info(
            f"=== Evaluation metrics ===\n"
            f"PESQ: {metrics_avg[0]:.4f},"
            f"SSNR: {metrics_avg[4]:.4f},"
            f"CSIG: {metrics_avg[1]:.4f},"
            f"CBAK: {metrics_avg[2]:.4f},"
            f"COVL: {metrics_avg[3]:.4f},"
            f"STOI: {metrics_avg[5]:.4f}\n"
        )
        return metrics_avg


    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        )
        noisy_dir = os.path.join(args.data_dir, "test/noisy")
        clean_dir = os.path.join(args.data_dir, "test/clean")
        total_start = time.time()
        for epoch in range(args.epochs):
            epoch_start = time.time()
            self.current_epoch = epoch
            self.model.train()
            self.discriminator.train()
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss = self.train_step(batch)
                if (step % args.log_interval) == 0:
                    template = "Epoch {}, Step {}, loss: {}, disc_loss: {}"
                    logging.info(template.format(epoch, step, loss, disc_loss))
            gen_loss = self.test()
            logging.info(f"\n=== Epoch {epoch} evaluation ===")
            epoch_end = time.time()
            print(f"Epoch {epoch+1}/{args.epochs} finished in {(epoch_end - epoch_start)/60:.2f} minutes")
            # print(noisy_dir, clean_dir)
            metrics = self.evaluate_epoch(noisy_dir, clean_dir)
            if epoch % 1 == 0 or epoch == args.epochs - 1:
                path = os.path.join(
                    args.save_model_dir,
                    "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5] + "_" + str(metrics[0])[:5],
                )
                if not os.path.exists(args.save_model_dir):
                    os.makedirs(args.save_model_dir)
                torch.save(self.model.state_dict(), path)
            scheduler_G.step()
            scheduler_D.step()
        total_end = time.time()
        print(f"\nTotal training time: {(total_end - total_start)/3600:.2f} hours")


def main():
    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    train_ds, test_ds = dataloader.load_data(args.data_dir, args.batch_size, 2, args.cut_len)
    trainer = Trainer(train_ds, test_ds)
    trainer.train()


if __name__ == "__main__":
    main()
