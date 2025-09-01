from models.conformer import ConformerBlock
from models.utils import *
from models.attn_blocks import SEBlock,CBAM, AxialImageTransformer,Simam_module,ECAAttention,Seq2DAttnWrapper,CrissCrossAttention

class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

    
class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64,attn=None):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
        # --- add attention block
        if attn == 'se':
            self.attn = SEBlock(channels)
        elif attn == 'cbam':
            self.attn = CBAM(channels)
        elif attn == 'simam':
            self.attn = Simam_module(channels)
        elif attn == 'eca':
            self.attn = ECAAttention(channels-1)
        else:
            self.attn = None
        
        # self.se = SEBlock(channels)
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )

    def forward(self, x):
        # input: [B,3,T,F]
        x = self.conv_1(x)
        # --- add attention block
        if self.attn is not None:
            x = self.attn(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)

    def forward(self, x_in):
        b, c, t, f = x_in.size() # [B,64,T,F] 
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b*f, t, c) # [B,F,T,64] --> [B*F,T,64]
        x_t = self.time_conformer(x_t) + x_t # [B*F,T,64] --> [B*F,T,64]
        # [B*F,T,64] --> [B,T,F,C] --> [B,T,F,C]
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x_f = self.freq_conformer(x_f) + x_f # [B,T,F,C] --> [B,T,F,C]
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2) # [B,T,F,C] --> [B,C,T,F]
        return x_f

class AxialTSCB(nn.Module):
    def __init__(self, num_channel=64, depth=2, heads=8, reversible=True, use_pos_emb=False, t_max=None, f_max=None):
        super().__init__()
        
        axial_pos_shape = (t_max, f_max) if use_pos_emb and (t_max is not None and f_max is not None) else None
        self.axial = AxialImageTransformer(
            dim=num_channel,
            depth=depth,
            heads=heads,
            dim_heads=None,
            dim_index=1,              # [B, C, T, F]
            reversible=reversible,
            axial_pos_emb_shape=axial_pos_shape
        )

    def forward(self, x_in):
        # x_in: [B, C, T, F]
        return self.axial(x_in)

class ccaTSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(ccaTSCB, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)

        self.time_conformer.attn.fn = Seq2DAttnWrapper(
            num_channel, attn2d_ctor=lambda c: CrissCrossAttention(c)
        )
        self.freq_conformer.attn.fn = Seq2DAttnWrapper(
            num_channel, attn2d_ctor=lambda c: CrissCrossAttention(c)
        )
    
    def forward(self, x_in):                 # x_in: [B,C,T,F']
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)  # [B*F, T, C]
        self.time_conformer.attn.fn.set_hw(t, 1)                        # H=T, W=1
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)  # [B*T, F', C]
        self.freq_conformer.attn.fn.set_hw(f, 1)                                       # H=F', W=1
        x_f = self.freq_conformer(x_f) + x_f

        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f
    

    

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1,attn=None):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        # add SE
        # self.se = SEBlock(num_channel)
        if attn == 'se':
            self.attn = SEBlock(num_channel)
        elif attn == 'cbam':
            self.attn = CBAM(num_channel)
        elif attn == 'simam':
            self.attn = Simam_module(num_channel)
        elif attn == 'eca':
            self.attn = ECAAttention(num_channel-1)
        else:
            self.attn = None

        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)


    def forward(self, x):
        x = self.dense_block(x)
        # add attention block
        if self.attn is not None:
            x = self.attn(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1) # [B,64,T,F] --> [B,F,T,64] --> [B,F,T]
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1) # [B,1,T,F]


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64,attn=None):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        # add attention block
        if attn == 'se':
            self.attn = SEBlock(num_channel)
        elif attn == 'cbam':
            self.attn = CBAM(num_channel)
        elif attn == 'simam':
            self.attn = Simam_module(num_channel)
        elif attn == 'eca':
            self.attn = ECAAttention(num_channel-1)
        else:
            self.attn = None
        
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x) # [B,1,T,F] --> [B,64,T,F]
        # add attention block
        # x = self.se(x) # [B,64,T,F] --> [B,64,T,F]
        if self.attn is not None:
            x = self.attn(x)
        x = self.sub_pixel(x) # [B,64,T,F] --> [B,64,T,F]
        x = self.prelu(self.norm(x)) # [B,64,T,F] --> [B,64,T,F]
        x = self.conv(x) # [B,64,T,F] --> [B,2,T,F]
        return x


class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201,attn1=None,attn2="mhsa",axial_depth=2, axial_heads=8, axial_reversible=True,
                 axial_use_pos_emb=False, axial_t_max=None, axial_f_max=None):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel,attn=attn1)

        if attn2 == "mhsa":
            self.TSCB_1 = TSCB(num_channel=num_channel)
            self.TSCB_2 = TSCB(num_channel=num_channel)
            self.TSCB_3 = TSCB(num_channel=num_channel)
            self.TSCB_4 = TSCB(num_channel=num_channel)
        elif attn2 == "axial":
            self.TSCB_1 = AxialTSCB(num_channel, axial_depth, axial_heads, axial_reversible,
                                     axial_use_pos_emb, axial_t_max, axial_f_max)
            self.TSCB_2 = AxialTSCB(num_channel, axial_depth, axial_heads, axial_reversible,
                                     axial_use_pos_emb, axial_t_max, axial_f_max)
            self.TSCB_3 = AxialTSCB(num_channel, axial_depth, axial_heads, axial_reversible,
                                     axial_use_pos_emb, axial_t_max, axial_f_max)
            self.TSCB_4 = AxialTSCB(num_channel, axial_depth, axial_heads, axial_reversible,
                                     axial_use_pos_emb, axial_t_max, axial_f_max)
        elif attn2 == "cca":
            self.TSCB_1 = ccaTSCB(num_channel=num_channel)
            self.TSCB_2 = ccaTSCB(num_channel=num_channel)
            self.TSCB_3 = ccaTSCB(num_channel=num_channel)
            self.TSCB_4 = ccaTSCB(num_channel=num_channel)
        else:
            raise ValueError(f"Unknown attention_type")

        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1,attn=attn1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel,attn=attn1)
        
    def forward(self, x):
        # input:[B, 2, T, F], x[:, 0, :, :] shape [B,T,F] 
        mag = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1) # [B,1,T,F]
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1) # [B,1,T,F]
        x_in = torch.cat([mag, x], dim=1) # [B,1,T,F] and [B,2,T,F] concat --> [B,3,T,F]

        out_1 = self.dense_encoder(x_in) # [B,3,T,F] --> [B,64,T,F] 
        out_2 = self.TSCB_1(out_1) # [B,64,T,F] 
        out_3 = self.TSCB_2(out_2) # [B,64,T,F] 
        out_4 = self.TSCB_3(out_3) # [B,64,T,F] 
        out_5 = self.TSCB_4(out_4) # [B,64,T,F] 

        mask = self.mask_decoder(out_5) # [B,1,T,F]
        out_mag = mask * mag # [B,1,T,F]

        complex_out = self.complex_decoder(out_5) # [B,1,T,F] --> [B,2,T,F]
        mag_real = out_mag * torch.cos(noisy_phase) # [B,1,T,F]
        mag_imag = out_mag * torch.sin(noisy_phase) # [B,1,T,F]
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1) # [B,1,T,F]
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1) # [B,1,T,F]

        return final_real, final_imag 