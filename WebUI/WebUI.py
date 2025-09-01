import gradio as gr
import librosa
import numpy as np
from models.generator import TSCNet
from models.utils import to_complex,power_uncompress,power_compress
import scipy.signal as signal
import torch
import soundfile as sf
import os
import tempfile
import torchaudio
from glob import glob
import matplotlib.pyplot as plt
import subprocess
import warnings
warnings.filterwarnings("ignore")

class CMGAN:
    def __init__(self, ckpt_path, n_fft=400,cut_len=16000 * 16,device=None,attn1=None,attn2="mhsa"):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = TSCNet(num_channel=64, num_features=n_fft // 2 + 1, attn1=attn1,attn2=attn2).cuda()
        self.model.load_state_dict(torch.load(ckpt_path,weights_only=True))
        self.model.to(self.device).eval()
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.cut_len = cut_len


    @torch.no_grad()
    def enhance(self, audio_path):
        """
        Enhance a single audio file

        Parameters:
            audio_path: Input audio path

        Return values:
            enhanced_audio: Enhanced audio data
            sr: Sample rate
        """

        # Load audio
        noisy, sr = torchaudio.load(audio_path)
        assert sr == 16000
        noisy = noisy.to(self.device)

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy = torch.transpose(noisy, 0, 1)
        noisy = torch.transpose(noisy * c, 0, 1)

        length = noisy.size(-1)
        frame_num = int(np.ceil(length / self.hop))
        padded_len = frame_num * self.hop
        if padded_len > self.cut_len:
            batch_size = int(np.ceil(padded_len / self.cut_len))
            while 100 % batch_size != 0:
                batch_size += 1
            noisy = torch.reshape(noisy, (batch_size, -1))

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).cuda(),
            onesided=True,
            return_complex=True
        )

        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

        est_spec_uncompress = power_uncompress(est_real, est_imag)
        est_complex = to_complex(est_spec_uncompress)

        est_audio = torch.istft(
            est_complex,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).cuda(),
            onesided=True,
        )
        # est_audio = est_audio[:, :original_len]
        est_audio = est_audio / c
        # enhanced_segments.append(est_audio)

        # est_audio = torch.cat(est_audio, dim=-1)
        est_audio = torch.flatten(est_audio)[:length].cpu().numpy()

        return est_audio, sr
    
SAMPLE_RATE = 16000
model_map = {
    "CMGAN_pesq": CMGAN("./checkpoints/CMGAN_PESQ",attn1=None),
    "CMGAN_ssnr": CMGAN("./checkpoints/CMGAN_SSNR",attn1=None),
    "CMGAN_MR_STFT": CMGAN("./checkpoints/CMGAN_MR_STFT",attn1=None),
    "CMGAN_SE": CMGAN("./checkpoints/CMGAN_SE",attn1="se"),
    "CMGAN_ECA": CMGAN("./checkpoints/CMGAN_ECA",attn1="eca"),
    "CMGAN_SIMAM": CMGAN("./checkpoints/CMGAN_SIMAM",attn1="simam"),
    "CMGAN_CBAM": CMGAN("./checkpoints/CMGAN_CBAM",attn1="cbam"),
}

def save_audio(audio_array,sr=SAMPLE_RATE):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_array, sr)
        return f.name

def enhance_audio(model_name, audio_path):
    """
    model_name: Selected model name
    audio_path: Local wav path provided by gr.Audio(type=“filepath”) after upload
    Returns: Temporary file path of enhanced wav
    """
    if audio_path is None or not os.path.exists(audio_path):
        return None
    data, orig_sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if orig_sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=SAMPLE_RATE
        )

        wav_t = torch.from_numpy(data).float().unsqueeze(0)  # [1, T]
        data = resampler(wav_t).squeeze(0).numpy()
        tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_in.name, data, SAMPLE_RATE)
        audio_path = tmp_in.name
    enhancer = model_map[model_name]
    enhanced, sr = enhancer.enhance(audio_path)
    temp_path = save_audio(enhanced,sr)
    return temp_path

def resample_audio(audio, orig_sr, target_sr):
    if orig_sr != target_sr:
        num_samples = int(len(audio) * float(target_sr) / orig_sr)
        audio = signal.resample(audio, num_samples)
    return audio

def batch_enhance_audio(model_name: str, input_dir: str, output_dir: str) -> str:
    """
    Batch enhancement:
      - input_dir: Specify the path of the folder to be processed.
      - output_dir: Specify the path of the folder where the enhanced results will be saved.
    Return: The path of the packaged ZIP file.
    """
    # Verify input
    if not os.path.isdir(input_dir):
        raise gr.Error(f"Input folder does not exist:{input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    enhancer = model_map[model_name]
    # Supports wav/flac/mp3 and other formats
    exts = ("*.wav", "*.flac", "*.mp3")
    files = []
    for ext in exts:
        files += glob(os.path.join(input_dir, ext))
    if not files:
        raise gr.Error("No audio files were found in the input folder.")

    for in_path in files:
        name = os.path.basename(in_path)
        enhanced_np, sr = enhancer.enhance(in_path)
        out_path = os.path.join(output_dir, name)
        sf.write(out_path, enhanced_np, sr)

    # Pack the entire output folder
    # zip_path = shutil.make_archive(output_dir, 'zip', root_dir=output_dir)
    return f"Processing complete, all enhanced files have been saved to:\n{output_dir}"

def analyze_audio_single(audio_path: str):
    """
    Waveform and spectrogram of a single audio clip, returned as (waveform.png, spectrogram.png)
    """
    if audio_path is None:
        return None, None

    # Load audio
    y, sr = librosa.load(audio_path)

    # Generate waveform
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    wave_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig1.savefig(wave_png, bbox_inches='tight')
    plt.close(fig1)

    # Generate spectrogram
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=512,
                                   x_axis='time', y_axis='linear', ax=ax2)
    ax2.set_title('Spectrogram')
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    spec_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig2.savefig(spec_png, bbox_inches='tight')
    plt.close(fig2)

    return wave_png, spec_png

def enhance_video_audio_ffmpeg(model_name: str, video_path: str) -> str:
    """
    Use FFmpeg + CMGAN model to denoise the audio of the video.
    """
    if not os.path.exists(video_path):
        return None

    noisy_wav    = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    enhanced_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    output_mp4   = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    # 1. Extract and resample audio using FFmpeg
    cmd_extract = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        noisy_wav
    ]
    subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # 2. CMGAN Enhancement
    enhancer = model_map[model_name]
    enhanced_np, sr = enhancer.enhance(noisy_wav)
    sf.write(enhanced_wav, enhanced_np, sr)

    # 3. Use FFmpeg to composite the enhanced audio track back into the video
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", enhanced_wav,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_mp4
    ]
    subprocess.run(cmd_mux, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return output_mp4

with gr.Blocks() as demo:
    gr.Markdown("# Audio Enhancement Demo")
    with gr.Tab("Single file processing"):
        with gr.Column():
            inp_audio = gr.Audio(source="upload", type="filepath", label="Upload noise audio")
            send_to_a1 = gr.Button("Send to audio analysis")
        mdl1 = gr.Dropdown(list(model_map.keys()), value="CMGAN_pesq", label="Select a model")
        btn1 = gr.Button("Start enhancing",variant="primary")
        with gr.Column():
            out_audio = gr.Audio(type="filepath", samplerate=SAMPLE_RATE, label="Enhanced audio")
            send_to_a2 = gr.Button("Send to audio analysis")
        btn1.click(fn=enhance_audio,
                   inputs=[mdl1, inp_audio],
                   outputs=[out_audio],
                   show_progress="minimal")

    with gr.Tab("Batch processing"):
        mdl2      = gr.Dropdown(list(model_map.keys()), value="CMGAN_pesq", label="Select a model")
        in_dir    = gr.Textbox(label="Input folder path", placeholder="D:/data/noisy_wavs")
        out_dir   = gr.Textbox(label="Output folder path", placeholder="D:/data/enhanced_wavs")
        btn2      = gr.Button("Start enhancing")
        status  = gr.Textbox(label="Outcome", interactive=False)
        btn2.click(fn=batch_enhance_audio,
                   inputs=[mdl2, in_dir, out_dir],
                   outputs=[status])

    with gr.Tab("Audio analysis"):
        with gr.Row():
            # The first audio
            with gr.Column():
                inp1    = gr.Audio(source="upload", type="filepath", label="Upload Audio 1")
                btn1    = gr.Button("Analyse Audio 1")
                wave1   = gr.Image(type="filepath", label="Audio 1 Waveform")
                spec1   = gr.Image(type="filepath", label="Audio 1 Spectrogram")
            # The second audio
            with gr.Column():
                inp2    = gr.Audio(source="upload", type="filepath", label="Upload Audio 2")
                btn2    = gr.Button("Analyse Audio 2")
                wave2   = gr.Image(type="filepath", label="Audio 2 Waveform")
                spec2   = gr.Image(type="filepath", label="Audio 2 Spectrogram")

            # The second audio
            with gr.Column():
                inp3    = gr.Audio(source="upload", type="filepath", label="Upload Audio 3")
                btn3    = gr.Button("Analyse Audio 3")
                wave3   = gr.Image(type="filepath", label="Audio 3 Waveform")
                spec3   = gr.Image(type="filepath", label="Audio 3 Spectrogram")

        btn1.click(fn=analyze_audio_single, inputs=[inp1], outputs=[wave1, spec1])
        btn2.click(fn=analyze_audio_single, inputs=[inp2], outputs=[wave2, spec2])
        btn3.click(fn=analyze_audio_single, inputs=[inp3], outputs=[wave3, spec3])
        send_to_a1.click(fn=lambda x: x, inputs=[inp_audio], outputs=[inp1])
        send_to_a2.click(fn=lambda x: x, inputs=[out_audio], outputs=[inp2])

    with gr.Tab("Video"):
        video_in = gr.Video(source="upload", type="filepath", label="Upload video files")
        mdl_video = gr.Dropdown(list(model_map.keys()), value="CMGAN", label="Select a model")
        btn_video = gr.Button("Start processing the video")
        video_out = gr.Video(type="filepath", label="Enhanced video")

        btn_video.click(
            fn=enhance_video_audio_ffmpeg,
            inputs=[mdl_video, video_in],
            outputs=[video_out]
        )

if __name__ == "__main__":
    demo.launch(share=False,debug=True)
