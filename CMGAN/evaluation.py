import numpy as np
from models import generator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
import warnings
warnings.filterwarnings("ignore")
import csv
import time

def to_complex(tensor_2channel: torch.Tensor):
    # tensor_2channel should be [B, 2, F, T]
    assert tensor_2channel.size(1) == 2, "Channel dimension must be 2 for real/imag"
    tensor = tensor_2channel.permute(0, 2, 3, 1).contiguous()  
    return torch.view_as_complex(tensor)
    
@torch.no_grad()
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


def evaluation(model_path, noisy_dir, clean_dir, save_tracks, saved_dir,attn1=None):
    n_fft = 400
    # model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1).cuda()
    model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1,attn1=attn1).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    metrics_total = np.zeros(6)
    for audio in audio_list:
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)
        est_audio, length = enhance_one_track(
            model, noisy_path, saved_dir, 16000 * 16, n_fft, n_fft // 4, save_tracks
        )
        clean_audio, sr = sf.read(clean_path)
        # if sr != 16000:
        #     clean_audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
        #         torch.tensor(clean_audio,dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        #     sr = 16000
        assert sr == 16000
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print(
        "pesq: ",
        metrics_avg[0],
        "csig: ",
        metrics_avg[1],
        "cbak: ",
        metrics_avg[2],
        "covl: ",
        metrics_avg[3],
        "ssnr: ",
        metrics_avg[4],
        "stoi: ",
        metrics_avg[5],
    )
    print("\n")

def evaluation2(model_path, noisy_dir, clean_dir, save_tracks, saved_dir, attn1=None,attn2='mhsa',csv_path=None):
    """
    Evaluate each audio and write metrics to CSV
    """
    n_fft = 400
    model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1,attn1=attn1,
                                                                                attn2=attn2,
                                                                                axial_depth=2,
                                                                                axial_heads=8,
                                                                                axial_reversible=True,
                                                                                axial_use_pos_emb=False 
                                                                                ).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir, exist_ok=True)

    if csv_path is None or len(str(csv_path).strip()) == 0:
        csv_path = os.path.join(saved_dir, "metrics.csv")

    audio_list = natsorted(os.listdir(noisy_dir))
    num = len(audio_list)
    metrics_total = np.zeros(6, dtype=np.float64)
    total_infer_sec = 0.0

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "pesq", "csig", "cbak", "covl", "ssnr", "stoi", "infer_sec"])

        for audio in audio_list:
            noisy_path = os.path.join(noisy_dir, audio)
            clean_path = os.path.join(clean_dir, audio)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            est_audio, length = enhance_one_track(
                model, noisy_path, saved_dir, 16000 * 16, n_fft, n_fft // 4, save_tracks
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_sec = time.perf_counter() - t0
            total_infer_sec += infer_sec
            clean_audio, sr = sf.read(clean_path)
            if sr != 16000:
                clean_audio = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=16000
                )(torch.tensor(clean_audio, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                sr = 16000
            assert sr == 16000
            metrics = compute_metrics(clean_audio, est_audio, sr, 0)
            metrics = np.array(metrics, dtype=np.float64)
            metrics_total += metrics

            writer.writerow([audio] + [f"{m:.6f}" for m in metrics.tolist()] + [f"{infer_sec:.6f}"])

    metrics_avg = (metrics_total / max(1, num)).tolist()
    infer_avg = total_infer_sec / max(1, num)

    print(
        f"=== Evaluation metrics (avg over {num}) ===\n"
        f"PESQ: {metrics_avg[0]:.4f}\n"
        f"CSIG: {metrics_avg[1]:.4f}\n"
        f"CBAK: {metrics_avg[2]:.4f}\n"
        f"COVL: {metrics_avg[3]:.4f}\n"
        f"SSNR: {metrics_avg[4]:.4f}\n"
        f"STOI: {metrics_avg[5]:.4f}\n"
        f"Inference time (avg, s): {infer_avg:.6f}"
    )

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["AVERAGE"] + [f"{m:.6f}" for m in metrics_avg] + [f"{infer_avg:.6f}"])

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./best_ckpt/ckpt_80',
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, default='dir to your VCTK-DEMAND test dataset',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")
parser.add_argument("--attn1", type=str, default=None,
            help="attention block to use, options: 'se', 'cbam', 'simam', 'eca', or None")
args = parser.parse_args()

if __name__ == "__main__":
    noisy_dir = os.path.join(args.test_dir, "noisy")
    clean_dir = os.path.join(args.test_dir, "clean")
    evaluation(args.model_path, noisy_dir, clean_dir, args.save_tracks, args.save_dir,args.attn1)
    # evaluation2(args.model_path, noisy_dir, clean_dir, args.save_tracks, args.save_dir,args.attn1,csv_path=os.path.join(args.save_dir, "metrics.csv"))