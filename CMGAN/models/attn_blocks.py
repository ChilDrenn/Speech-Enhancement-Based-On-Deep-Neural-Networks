import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Args:
        in_channels (int): Number of input channels
        reduction (int): Channel reduction ratio for the bottleneck. Default: 16
    """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=[2, 3])  # [b, c]
        y = self.se(y).view(b, c, 1, 1)
        return x * y   # Channel-wise multiplication
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
        in_planes (int): Number of input channels
        ratio (int): Reduction ratio for bottleneck. Default: 16
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)  

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  
        out = avg_out + max_out  
        return self.sigmoid(out)  


class SpatialAttention(nn.Module):
    """
    Args:
        kernel_size (int): Size of conv kernel. Must be 3 or 7
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  
        padding = 3 if kernel_size == 7 else 1  

        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True) 
        x = torch.cat([avg_out, max_out], dim=1)  
        x = self.conv1(x)  
        return self.sigmoid(x)  


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  
        self.sa = SpatialAttention(kernel_size)  

    def forward(self, x):
        out = x * self.ca(x)  
        result = out * self.sa(out)  
        return result 

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

class ECAAttention(nn.Module):
    """
    Args:
        kernel_size (int): Size of the 1D conv kernel. Default: 3
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()  

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  
                if m.bias is not None:
                    init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  
                init.constant_(m.bias, 0)  
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001) 
                if m.bias is not None:
                    init.constant_(m.bias, 0)  


    def forward(self, x):
        y = self.gap(x)  
        y = y.squeeze(-1).permute(0, 2, 1) 
        y = self.conv(y)  
        y = self.sigmoid(y)  
        y = y.permute(0, 2, 1).unsqueeze(-1) 
        return x * y.expand_as(x)  

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

class Simam_module(torch.nn.Module):
    """
    Args:
        e_lambda (float): Small constant for numerical stability. Default: 1e-4
    """
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid() 
        self.e_lambda = e_lambda  

    def forward(self, x):
        b, c, h, w = x.size() 
        n = w * h - 1  
        
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        return x * self.act(y)

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net 
        self.cpu_state = None   
        self.cuda_in_fwd = None 
        self.gpu_devices = None  
        self.gpu_states = None 
	
    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f) 
        self.g = Deterministic(g)  

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=1) 
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)  
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args) 

        return torch.cat([y1, y2], dim=1) 

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=1)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=1)
            dx = torch.cat([dx1, dx2], dim=1)

        return x, dx


class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = F
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=1)

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    def __init__(self, blocks, ):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])

    def forward(self, x, arg_route=(True, True), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}
        x = torch.cat((x, x), dim=1)  
        x = _ReversibleFunction.apply(x, self.blocks, block_kwargs)
        return torch.stack(x.chunk(2, dim=1)).mean(dim=0)


def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)  
    arr = sorted(arr) 
    return map_el_ind(arr, 0), map_el_ind(arr, 1) 



def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x

class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

      
        axial = axial.reshape(-1, t, d)

        
        axial = self.fn(axial, **kwargs)

       
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial



class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class AxialAttention(nn.Module):
    """
    Args:
        dim (int): Number of input channels
        num_dimensions (int): Number of spatial dimensions (default: 2 for images)
        heads (int): Number of attention heads (default: 8)
        dim_heads (int, optional): Dimension of each attention head
        dim_index (int): Dimension to apply attention to (-1 for last dim)
        sum_axial_out (bool): Whether to sum or chain attention outputs
    """
    def __init__(self, dim, num_dimensions=2, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out


class AxialImageTransformer(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_heads=None, dim_index=1, reversible=True, axial_pos_emb_shape=None):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)

        get_ff = lambda: nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim, 3, padding=1)
        )

        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(
            axial_pos_emb_shape) else nn.Identity()

        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_functions = nn.ModuleList(
                [PermuteToFrom(permutation, PreNorm(dim, SelfAttention(dim, heads, dim_heads))) for permutation in
                 permutations])
            conv_functions = nn.ModuleList([get_ff(), get_ff()])
            layers.append(attn_functions)
            layers.append(conv_functions)

        execute_type = ReversibleSequence if reversible else Sequential
        self.layers = execute_type(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        return self.layers(x)

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

def neg_inf_diag_like(x, H, W):
    
    B = x.shape[0]
    device, dtype = x.device, x.dtype
    mask = torch.full((H, H), float('-inf'), device=device, dtype=dtype)
    mask = torch.diag(mask.diag())  
    return mask.unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    """
    Implements the CCA module that decomposes 2D attention into two 1D attentions for efficiency.
    """
    def __init__(self, in_channels):
        super().__init__()
        c = in_channels
        mid = max(1, c // 8)
        self.q = nn.Conv2d(c, mid, 1, bias=False)
        self.k = nn.Conv2d(c, mid, 1, bias=False)
        self.v = nn.Conv2d(c, c,   1, bias=False)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        # Q, K, V
        q = self.q(x); k = self.k(x); v = self.v(x)

        qH = q.permute(0,3,1,2).contiguous().view(B*W, C//8 if C>=8 else q.size(1), H).permute(0,2,1)
        kH = k.permute(0,3,1,2).contiguous().view(B*W, -1, H)
        vH = v.permute(0,3,1,2).contiguous().view(B*W, C, H)

        qW = q.permute(0,2,1,3).contiguous().view(B*H, q.size(1), W).permute(0,2,1)
        kW = k.permute(0,2,1,3).contiguous().view(B*H, -1, W)
        vW = v.permute(0,2,1,3).contiguous().view(B*H, C, W)

        eH = torch.bmm(qH, kH) + neg_inf_diag_like(qH, H, W)          # [B*W, H, H]
        eH = eH.view(B, W, H, H).permute(0,2,1,3)                      # [B, H, W, H]
        eW = torch.bmm(qW, kW).view(B, H, W, W)                        # [B, H, W, W]

        attn = self.softmax(torch.cat([eH, eW], dim=3))
        aH = attn[:,:,:,:H].permute(0,2,1,3).contiguous().view(B*W, H, H)   # [B*W,H,H]
        aW = attn[:,:,:,H:].contiguous().view(B*H, W, W)                     # [B*H,W,W]

        outH = torch.bmm(vH, aH.transpose(1,2)).view(B, W, C, H).permute(0,2,3,1)  # [B,C,H,W]
        outW = torch.bmm(vW, aW.transpose(1,2)).view(B, H, C, W).permute(0,2,1,3)  # [B,C,H,W]
        return self.gamma * (outH + outW) + x

class Seq2DAttnWrapper(nn.Module):

    def __init__(self, dim, attn2d_ctor):
        super().__init__()
        self.dim = dim
        self.attn2d = attn2d_ctor(dim)
        self.H = None; self.W = None

    @torch.no_grad()
    def set_hw(self, H, W): 
        self.H, self.W = int(H), int(W)

    def forward(self, x, mask=None):
        assert self.H is not None and self.W is not None, "call set_hw(H,W) before forward"
        B, N, C = x.shape
        assert N == self.H * self.W and C == self.dim
        x2d = x.transpose(1, 2).contiguous().view(B, C, self.H, self.W)   # [B,C,H,W]
        x2d = self.attn2d(x2d)                                            # CCA
        x   = x2d.view(B, C, -1).transpose(1, 2).contiguous()             # [B,N,C]
        return x