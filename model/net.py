from functools import partial
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from model.dmsp import DMSP, apply_dmsp
from model.msa import MSA, apply_msa
from sam2.build_sam import build_sam2
from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock
import torch
from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding




class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class Decoder(Module):
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder, self).__init__()
        self.timeattn1 = MCA(cond_dim=dims[0], time_dim=dims[0], num_heads=12)
        self.timeattn2 = MCA(cond_dim=dims[1], time_dim=dims[1], num_heads=12)
        self.timeattn3 = MCA(cond_dim=dims[2], time_dim=dims[2], num_heads=12)
        self.timeattn4 = MCA(cond_dim=dims[3], time_dim=dims[3], num_heads=12)
        self.timelinear1 = nn.Linear(in_features=dims[0], out_features=dims[1])
        self.timelinear2 = nn.Linear(in_features=dims[1], out_features=dims[2])
        self.timelinear3 = nn.Linear(in_features=dims[2], out_features=dims[3])


        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim
        
        self.conv_c1 = nn.Conv2d(c1_in_channels, 1, kernel_size=1)
        self.conv_c2 = nn.Conv2d(c2_in_channels, 1, kernel_size=1)

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.final_conv = nn.Sequential(
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim * 2, kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim * 4, kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim * 8, kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )
        
        self.fft = BlockFFT(dim=c4_in_channels, h=11, w=11)

        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        resnet_block = partial(ResnetBlock, groups=8)

        self.down = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )

        self.down_2x_1 = nn.Sequential(
            ConvModule(embedding_dim, embedding_dim, kernel_size=3, padding=1, stride=2, norm_cfg=dict(type='BN')),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(embedding_dim, embedding_dim*2, kernel_size=3, padding=1, norm_cfg=dict(type='BN'))
        )

        self.down_2x_2 = nn.Sequential(
            ConvModule(embedding_dim*2, embedding_dim*4, kernel_size=3, padding=1, stride=2, norm_cfg=dict(type='BN')),
            ConvModule(embedding_dim*4, embedding_dim*4, kernel_size=3, padding=1, norm_cfg=dict(type='BN'))
        )


        self.down_2x_3 = nn.Sequential(
            ConvModule(embedding_dim*4, embedding_dim*8, kernel_size=3, stride=2, padding=1, norm_cfg=dict(type='BN')),
            ConvModule(embedding_dim*8, embedding_dim*8, kernel_size=3, padding=1, norm_cfg=dict(type='BN'))
        )
        self.cat_conv = ConvModule(in_channels=embedding_dim * 16, out_channels=embedding_dim*8, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))

        self.up_2x_3 = nn.Sequential(
            ConvModule(in_channels=embedding_dim*8, out_channels=embedding_dim*4, kernel_size=1,
                       norm_cfg=dict(type='BN')),
            Upsample(embedding_dim*4, embedding_dim*4, factor=2),
            ConvModule(in_channels=embedding_dim*4, out_channels=embedding_dim*4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN')),
        )
        self.up_2x_2 = nn.Sequential(
            ConvModule(in_channels=embedding_dim*4, out_channels=embedding_dim*2, kernel_size=1,
                       norm_cfg=dict(type='BN')),
            Upsample(embedding_dim*2, embedding_dim*2, factor=2),
            ConvModule(in_channels=embedding_dim*2, out_channels=embedding_dim*2, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN')),
        )
        self.up_2x_1 = nn.Sequential(
            ConvModule(in_channels=embedding_dim*2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN')),
            Upsample(embedding_dim, embedding_dim, factor=2),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN')),
        )

        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),

            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )

        self.pred = nn.Sequential(


            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )
        self.cond_attention1 = MultiScaleSpatialAttention(c1_in_channels)
        self.cond_attention2 = MultiScaleSpatialAttention(c2_in_channels)
        self.cond_attention3 = MultiScaleSpatialAttention(c3_in_channels)
        self.dmsp = nn.ModuleList(DMSP(channels) for channels in dims)
        self.msa = nn.ModuleList(
            MSA(channels, in_size=patch, patch=patch, mem_slots=slots, topk=topk)
            for channels, patch, slots, topk in (
                (dims[0], 16, 256, 8),
                (dims[1], 16, 128, 8),
                (dims[2], 8, 64, 8),
                (dims[3], 4, 32, 4),
            )
        )
        self.cond_fusion = ProgressiveFusion(dims)
        self.feature_fusion = feature_fusion(channels=c4_in_channels, t_dim=embedding_dim)
    def forward(self, inputs, timesteps, x):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
        t1 = self.timelinear1(t)
        t2 = self.timelinear2(t1)
        t3 = self.timelinear3(t2)
        c1, c2, c3, c4 = inputs

        c1 = self.timeattn1(c1, t)
        c2 = self.timeattn2(c2, t1)
        c3 = self.timeattn3(c3, t2)
        c4 = self.timeattn4(c4, t3)
        c1, c2, c3, c4 = (
            apply_msa(msa, apply_dmsp(dmsp, feature))
            for msa, dmsp, feature in zip(self.msa, self.dmsp, (c1, c2, c3, c4))
        )
        _x = [x]
        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)
        c1_attention = self.cond_attention1(c1)

        x = x + x*c1_attention

        for blk in self.down_2x_1:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)
        c2_attention = self.cond_attention2(c2)
        x = x + x*c2_attention     

        for blk in self.down_2x_2:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)
        c3_attention = self.cond_attention3(c3)
        x = x + x*c3_attention
        for blk in self.down_2x_3:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)

        c_final = self.cond_fusion(c1, c2, c3, c4)


        x = self.fft(x)

        x = self.feature_fusion(x, c_final, t)
        
        for blk in self.up_2x_3:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        
        for blk in self.up_2x_2:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        
        for blk in self.up_2x_1:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        
        for blk in self.up:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)

        x = self.pred(x)
        return x


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        output = self.block(promped)
        return output
    
class SAM2backbone(nn.Module):
    def __init__(self, checkpoint_path=None, freeze_backbone=True, image_size=352,
                 model_cfg="sam2_hiera_l.yaml") -> None:
        super(SAM2backbone, self).__init__()    
        model = build_sam2(model_cfg, device="cpu")
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        if checkpoint_path:
            self._load_encoder_checkpoint(checkpoint_path)

        for param in self.encoder.parameters():
            param.requires_grad = not freeze_backbone
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        self.proj = nn.Conv2d(3, 3, kernel_size=3, stride=1,
                              padding=(1, 1))
        self.x_proj = nn.Conv2d(1, 3, kernel_size=3, stride=1,
                                       padding=(1, 1))

        self.x_proj.weight.data.zero_()
        self.x_proj.bias.data.zero_()
        self.time_embed_dim = self.encoder.channel_list[-1]
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )
        self.time_processor = nn.Sequential(
            nn.Linear(self.time_embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256 * 16 * 16),
            nn.Unflatten(1, (256, 16, 16)),


            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Upsample(size=(image_size, image_size), mode='bilinear'),
            nn.Sigmoid()
        )
    def _load_encoder_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        prefix = "image_encoder.trunk."
        trunk_state = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
        if not trunk_state:
            raise RuntimeError(f"No {prefix} weights found in {checkpoint_path}")
        self.encoder.load_state_dict(trunk_state, strict=True)

    def forward(self, x, timesteps, cond_img):

        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
        t = self.time_processor(t)
        cond_img = cond_img + x*t
        x1, x2, x3, x4 = self.encoder(cond_img)
        return x1, x2, x3, x4


class BlockFFT(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(dim, h, w//2 + 1, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        input_dtype = x.dtype
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x.float(), dim=(2, 3), norm='ortho')
        x = x * torch.view_as_complex(self.complex_weight)
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        return x.reshape(B, C, H, W).to(input_dtype)


class MCA(nn.Module):
    """Multi-head Cross Attention for diffusion timestep conditioning."""
    def __init__(self, cond_dim, time_dim=144, num_heads=12, residual=True):
        super().__init__()
        assert time_dim % num_heads == 0, "time_dim必须能被num_heads整除"
        self.num_heads = num_heads
        self.head_dim = time_dim // num_heads
        self.residual = residual


        self.key = nn.Conv2d(cond_dim, time_dim, kernel_size=1)
        self.value = nn.Conv2d(cond_dim, time_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(time_dim, cond_dim, kernel_size=1)

    def forward(self, cond, t):
        B, C, H, W = cond.shape
        residual = cond


        q = t.view(B, self.num_heads, self.head_dim).unsqueeze(-1)


        k = self.key(cond).view(B, self.num_heads, self.head_dim, H*W)
        v = self.value(cond).view(B, self.num_heads, self.head_dim, H*W)


        scale = self.head_dim ** 0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) / scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)


        out = out.view(B, -1, 1, 1)
        out = self.out_proj(out).expand_as(cond)
        
        return out + residual if self.residual else out


class EnhancedChannelAttention(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        输入: 
            x : 特征图 [B, C, H, W]
        输出: 
            weights : 通道权重 [B, C, 1, 1]
        """
        b, c, _, _ = x.size()


        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(b, c)


        max_out = self.max_pool(x)
        max_out = max_out.view(b, c)


        combined = avg_out + max_out


        channel_weights = self.fc(combined)
        channel_weights = channel_weights.view(b, c, 1, 1)
        
        return channel_weights

class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernels=[1,3,5]):
        super().__init__()


        self.branches = nn.ModuleList([
            nn.Sequential(

                nn.Conv2d(in_channels, in_channels, k, padding=k//2, bias=False),

                SpatialAttention()
            ) for k in kernels
        ])
        
    def forward(self, x):

        branch_outs = [branch(x) for branch in self.branches]


        summed_attn = torch.stack(branch_outs).sum(dim=0)
        return summed_attn

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)


        attn = self.spatial_attn(combined)
        return attn


class feature_fusion(nn.Module):
    def __init__(self, channels=1152, t_dim=144):
        super().__init__()

        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, 11*11),
            nn.GELU(),
            nn.Unflatten(1, (1, 11, 11)),
            nn.Sigmoid()
        )

        self.channel_attn = EnhancedChannelAttention(channels)


        self.norm = nn.Sequential(
            nn.LayerNorm([channels, 11, 11]),
            nn.Sigmoid()
        )
        self.cat_conv = nn.Conv2d(2*channels, channels, 3, padding=1)
        self.cat_conv1 = nn.Conv2d(2*channels, channels, 3, padding=1)
        self.spatial_attention = MultiScaleSpatialAttention(channels)
    def forward(self, x, c, t):
        """
        输入:
        x: 扩散特征 [B,1152,11,11] 
        c: 条件特征 [B,1152,11,11]
        t: 时间步嵌入 [B,t_dim]
        """


        multiply = x * c


        spatialmap = self.spatial_attention(multiply)


        x_enhanced = x + x * spatialmap
        c_enhanced = c + c * spatialmap


        combined = torch.cat([x_enhanced, c_enhanced], dim=1)
        combined = self.cat_conv(combined)

        ch_attn = self.channel_attn(combined)


        x_modulated = x_enhanced * ch_attn
        c_modulated = c_enhanced * ch_attn


        t_gate = self.t_proj(t)
        cat_fusion = self.cat_conv1(torch.cat([x_modulated, c_modulated], dim=1))

        fused = cat_fusion + t_gate*x_modulated


        return fused


class ProgressiveFusion(nn.Module):
    def __init__(self, dims=(144, 288, 576, 1152)):
        super().__init__()
        c1, c2, c3, c4 = dims


        self.stage1 = FusionBlock(in_ch_low=c1, in_ch_high=c2, out_ch=c2)


        self.stage2 = FusionBlock(in_ch_low=c2, in_ch_high=c3, out_ch=c3)


        self.stage3 = FusionBlock(in_ch_low=c3, in_ch_high=c4, out_ch=c4)
        
        self.convstage1 = ConvFusionBlock(in_ch_low=c2, in_ch_high=c3, out_ch=c3)

        self.convstage2 = ConvFusionBlock(in_ch_low=c3, in_ch_high=c4, out_ch=c4)
    def forward(self, f1, f2, f3, f4):


        fused_12 = self.stage1(f1, f2)


        fused_23 = self.stage2(f2, f3)


        fused_34 = self.stage3(f3, f4)

        fused_123 = self.convstage1(fused_12, fused_23)
        output = self.convstage2(fused_123, fused_34)        
        return output


class FusionBlock(nn.Module):
    """ 相邻尺度融合单元（低分辨率特征 + 高分辨率特征） """
    def __init__(self, in_ch_low, in_ch_high, out_ch):
        super().__init__()

        self.process_high = nn.Sequential(
            nn.Conv2d(in_ch_low, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )


        self.process_low = nn.Sequential(
            nn.Conv2d(in_ch_high, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )


        self.combinefusion = nn.Conv2d(2*out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        
    def forward(self, feat_high, feat_low):
        """
        输入:
            feat_low: 低分辨率特征 [B, in_ch_low, H, W]
            feat_high: 高分辨率特征 [B, in_ch_high, H, W]
        输出:
            融合后的特征 [B, out_ch, H/2, W/2]
        """

        low_proj = self.process_low(feat_low)  


        high_proj = self.process_high(feat_high) 
        
        addfusion = low_proj+high_proj
        multipfusion = low_proj*high_proj


        combined = torch.cat([addfusion, multipfusion], dim=1)


        fused = self.combinefusion(combined)
        return fused
    
class ConvFusionBlock(nn.Module):
    """ 相邻尺度融合单元（低分辨率特征 + 高分辨率特征） """
    def __init__(self, in_ch_low, in_ch_high, out_ch):
        super().__init__()

        self.process_high = nn.Sequential(
            nn.Conv2d(in_ch_low, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )


        self.process_low = nn.Sequential(
            nn.Conv2d(in_ch_high, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )


        self.combinefusion = nn.Conv2d(2*out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        
    def forward(self, feat_high, feat_low):
        """
        输入:
            feat_low: 低分辨率特征 [B, in_ch_low, H, W]
            feat_high: 高分辨率特征 [B, in_ch_high, H, W]
        输出:
            融合后的特征 [B, out_ch, H/2, W/2]
        """

        low_proj = self.process_low(feat_low)  


        high_proj = self.process_high(feat_high) 


        combined = torch.cat([low_proj, high_proj], dim=1)


        fused = self.combinefusion(combined)
        return fused


# Legacy import compatibility; parameter names remain unchanged.
TimeCrossAttn = MCA
