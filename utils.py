import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def to_complex(tensor_2channel: torch.Tensor):
    # tensor_2channel should be [B, 2, F, T]
    assert tensor_2channel.size(1) == 2, "Channel dimension must be 2 for real/imag"
    tensor = tensor_2channel.permute(0, 2, 3, 1).contiguous()
    return torch.view_as_complex(tensor)


# def power_compress(x):
#     real = x[..., 0]
#     imag = x[..., 1]
#     spec = torch.complex(real, imag)
#     mag = torch.abs(spec)
#     phase = torch.angle(spec)
#     mag = mag**0.3
#     real_compress = mag * torch.cos(phase)
#     imag_compress = mag * torch.sin(phase)
#     return torch.stack([real_compress, imag_compress], 1)

def power_compress(x: torch.Tensor):
    """
    x: complex tensor of shape [B, F, T]
    returns: real tensor of shape [B, 2, F, T]
    """

    # mag = torch.abs(x) ** 0.3
    # phase = torch.angle(x)
    mag = x.abs()
    phase = x.angle()
    mag = mag ** 0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], dim=1)


def power_uncompress(real, imag):
    real = real.squeeze(1)  # [B, F, T]
    imag = imag.squeeze(1)
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    # return torch.stack([real_compress, imag_compress], dim=1)  # [B, 2, F, T]
    return torch.stack([real_compress, imag_compress], 1)


# def power_uncompress(real, imag):
#     real = real.squeeze(1)  # [B, F, T]
#     imag = imag.squeeze(1)
#     spec = torch.complex(real, imag)
#     mag = torch.abs(spec)
#     phase = torch.angle(spec)
#     mag = mag ** (1.0 / 0.3)
#     real_compress = mag * torch.cos(phase)
#     imag_compress = mag * torch.sin(phase)
#     return torch.stack([real_compress, imag_compress], -1)

def multi_res_stft_loss(x, y, fft_sizes=[256, 512, 1024], hops=[64, 128, 256]):
    loss = 0.0
    for n_fft, hop in zip(fft_sizes, hops):
        X = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft).cuda(), return_complex=True)
        Y = torch.stft(y, n_fft, hop, window=torch.hann_window(n_fft).cuda(), return_complex=True)
        loss += F.l1_loss(X.abs(), Y.abs())
    return loss / len(fft_sizes)


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
