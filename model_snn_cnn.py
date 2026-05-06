import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from einops import rearrange, repeat
import math

# ======== ✅ Optimized HDREnhancer (for [0,1] input, RGB only) ========
class HDREnhancer(nn.Module):
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha

        # Channel attention: small MLP → [B,3,H,W] → [B,3,1,1]
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 1),
            nn.Sigmoid()
        )

        # Spatial attention via Sobel edge maps
        self.spatial_att = nn.Sequential(
            nn.Conv2d(6, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # Fixed Sobel kernels (repeated for 3 channels, groups=3 in conv)
        sobel_x = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1))  # (3,1,3,3)
        self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))

    def forward(self, x):
        # x: [B, 3, T, H, W] or [B, 3, H, W]
        is_5d = x.dim() == 5
        if is_5d:
            B, C, T, H, W = x.shape
            x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, 3, H, W)
        else:
            x_flat = x  # (B, 3, H, W)

        # Channel attention
        ca = self.channel_att(x_flat)      # (B*T, 3, 1, 1)
        x_ca = x_flat * ca                 # channel-wise reweight

        # Spatial attention (Sobel edge enhancement)
        gx = F.conv2d(x_flat, self.sobel_x, padding=1, groups=3)  # (B*T, 3, H, W)
        gy = F.conv2d(x_flat, self.sobel_y, padding=1, groups=3)
        sa = self.spatial_att(torch.cat([gx, gy], dim=1))         # (B*T, 1, H, W)

        # Enhanced output: x * ca * (1 + alpha * sa)
        x_enh = x_ca * (1.0 + self.alpha * sa)

        # Reshape back if 5D
        if is_5d:
            x_enh = x_enh.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W)
        return x_enh


# ======== MSC Module (unchanged) ========
class MSC(nn.Module):
    def __init__(self, dim, num_heads=8, topk=True, kernel=[3,5,7], s=[1,1,1], pad=[1,2,3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3):
        super(MSC, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q    = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv   = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.k1   = k1
        self.k2   = k2
        
        self.attn1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2], stride=s[2], padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)
        self.topk = topk

    def forward(self, x, y):
        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        y = y1 + y2 + y3
        y = y.flatten(-2, -1)
        y = y.transpose(1, 2)
        y = self.layer_norm(y)

        x = rearrange(x, 'b c h w -> b (h w) c')
        B, N1, C = y.shape
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index1 = torch.topk(attn, k=int(N1 // self.k1), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index1, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = attn1 @ v

        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index2 = torch.topk(attn, k=int(N1 // self.k2), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index2, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = attn2 @ v

        out = out1 * self.attn1 + out2 * self.attn2
        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        hw = int(math.sqrt(N))
        x = rearrange(x, 'b (h w) c -> b c h w', h=hw, w=hw)
        return x


# ======== MFCM Module (unchanged) ========
class MFCM(nn.Module):
    def __init__(self, in_channels=[128, 256, 512], align_dim=128, num_heads=4, k1=2, k2=3):
        super(MFCM, self).__init__()
        self.align_dim = align_dim

        self.align2 = nn.Sequential(
            nn.Conv2d(in_channels[0], align_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(align_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(align_dim, align_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(align_dim),
            nn.ReLU(inplace=True)
        )
        self.align3 = nn.Sequential(
            nn.Conv2d(in_channels[1], align_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(align_dim),
            nn.ReLU(inplace=True)
        )
        self.align4 = nn.Sequential(
            nn.Conv2d(in_channels[2], align_dim, 1),
            nn.BatchNorm2d(align_dim),
            nn.ReLU(inplace=True)
        )

        self.msc2 = MSC(dim=align_dim, num_heads=num_heads, k1=k1, k2=k2)
        self.msc3 = MSC(dim=align_dim, num_heads=num_heads, k1=k1, k2=k2)

    def forward(self, feats):
        f2, f3, f4 = feats
        a2 = self.align2(f2)
        a3 = self.align3(f3)
        a4 = self.align4(f4)

        o2 = self.msc2(a4, a2)
        o3 = self.msc3(a4, a3)

        fused = torch.cat([o2, o3, a4, a4], dim=1)
        return fused


# ======== LinearTransformLayer & SS2D (unchanged) ========
class LinearTransformLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        return x * self.gamma + self.beta


class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1, groups=self.d_inner)
        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False).weight
            for _ in range(4)
        ]))
        self.dt_projs_weight = nn.Parameter(torch.stack([
            self.dt_init(self.dt_rank, self.d_inner) for _ in range(4)
        ]))
        self.dt_projs_bias = nn.Parameter(torch.stack([
            self.dt_init_bias(self.dt_rank, self.d_inner) for _ in range(4)
        ]))
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4)
        self.Ds = self.D_init(self.d_inner, copies=4)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_min=0.001, dt_max=0.1):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj.weight

    @staticmethod
    def dt_init_bias(dt_rank, d_inner, dt_min=0.001, dt_max=0.1):
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        return inv_dt

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner)
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies).flatten(0, 1)
        return nn.Parameter(A_log)

    @staticmethod
    def D_init(d_inner, copies=1):
        D = torch.ones(d_inner)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies).flatten(0, 1)
        return nn.Parameter(D)

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([
            x.view(B, -1, L),
            x.transpose(-2, -1).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        # Note: selective_scan_fn requires mamba_ssm; if unavailable, replace with dummy or remove SS2D
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            out_y = selective_scan_fn(
                xs, dts, As, Bs, Cs, Ds,
                delta_bias=dt_bias,
                delta_softplus=True
            ).view(B, K, -1, L)
        except ImportError:
            # Fallback: identity (remove CMM if mamba not available)
            out_y = xs.view(B, K, -1, L)

        y1 = out_y[:, 0]
        y2 = out_y[:, 1]
        y3 = torch.flip(out_y[:, 2], dims=[-1])
        y4 = torch.flip(out_y[:, 3], dims=[-1]).transpose(-2, -1).contiguous().view(B, -1, L)
        y = y1 + y2 + y3 + y4
        return y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        return self.out_proj(y)


# ======== CMMBlock (unchanged) ========
class CMMBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.trans_rgb = LinearTransformLayer(in_channels)
        self.trans_evt = LinearTransformLayer(in_channels)
        self.mamba = SS2D(d_model=in_channels)

    def forward(self, f_rgb, f_evt):
        z_rgb = self.trans_rgb(f_rgb)
        z_evt = self.trans_evt(f_evt)

        B, C, H, W = z_rgb.shape
        merged = torch.stack([z_evt, z_rgb], dim=-1)
        seq = merged.view(B, C, H, W * 2).permute(0, 2, 3, 1)
        fused = self.mamba(seq)
        fused = fused.permute(0, 3, 1, 2).view(B, C, H, W, 2)

        delta_evt = fused[..., 0]
        delta_rgb = fused[..., 1]

        return f_rgb + delta_rgb, f_evt + delta_evt


# ======== ConvLSTM (lightweight) ========
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels=None, kernel_size=3, num_layers=1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels or input_channels
        self.num_layers = num_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = self.input_channels if i == 0 else self.hidden_channels
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_channels, kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        h = [
            torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            for _ in range(self.num_layers)
        ]

        for t in range(T):
            inp = x[:, t]
            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.cell_list[layer_idx](
                    inp if layer_idx == 0 else h[layer_idx - 1],
                    (h[layer_idx], c[layer_idx])
                )
        return h[-1]  # [B, hid, H, W]


# ✅ NEW: ASFM — Faithful to paper (Fig.4 + Sec.2.3)
class ASFM(nn.Module):
    def __init__(self, in_channels, reduction=16, alpha_amp=1.0, alpha_low=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.alpha_amp = alpha_amp
        self.alpha_low = alpha_low
        mid_channels = in_channels // reduction

        # Shared 1×1 conv for channel reduction (Fig.4 left)
        self.conv_reduce = nn.Conv2d(in_channels, mid_channels, 1, bias=False)

        # Unified weight generation after merging (Eq.3–4)
        self.weight_net = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels * 2, 1, bias=False),
            nn.Sigmoid()  # bounded [0,1], enables meaningful ranking
        )

        # Restore conv: cat(enh_e, enh_f) → C
        self.restore_conv = nn.Conv2d(2 * in_channels, in_channels, 1, bias=False)

    def forward(self, fe: torch.Tensor, ff: torch.Tensor):
        """
        fe, ff: [B, C=512, H, W] — outputs of CMM
        returns: [B, C=512, H, W]
        """
        B, C, H, W = fe.shape
        mid_C = C // self.reduction  # e.g. 512//16=32

        # ↓ Step 1: Shared reduction
        fe_red = self.conv_reduce(fe)   # [B, mid_C, H, W]
        ff_red = self.conv_reduce(ff)

        # ↓ Step 2: Global pooling (avg + max → richer stats)
        fe_pool = F.adaptive_avg_pool2d(fe_red, 1) + F.adaptive_max_pool2d(fe_red, 1)  # [B, mid_C, 1, 1]
        ff_pool = F.adaptive_avg_pool2d(ff_red, 1) + F.adaptive_max_pool2d(ff_red, 1)

        # ↓ Step 3: MERGE → generate unified β_M (Eq.3–4)
        merged_pool = torch.cat([fe_pool, ff_pool], dim=1)  # [B, 2*mid_C, 1, 1]
        beta = self.weight_net(merged_pool).squeeze(-1).squeeze(-1)  # [B, 2*mid_C]

        # ↓ Step 4: Median-based ranking (Eq.5)
        D = torch.median(beta, dim=1, keepdim=True).values  # [B, 1]
        beta_tilde = torch.where(beta >= D,
                                 beta * self.alpha_amp,
                                 beta * self.alpha_low)  # α_low=0 → suppress weak channels

        # ↓ Step 5: Split → upscale to full C (Eq.6)
        w_e_mid = beta_tilde[:, :mid_C].unsqueeze(-1).unsqueeze(-1)  # [B, mid_C, 1, 1]
        w_f_mid = beta_tilde[:, mid_C:].unsqueeze(-1).unsqueeze(-1)

        w_e = w_e_mid.repeat_interleave(self.reduction, dim=1)  # [B, C, 1, 1]
        w_f = w_f_mid.repeat_interleave(self.reduction, dim=1)

        # ↓ Step 6: Reweight CMM-enhanced features
        fe_weighted = fe * w_e
        ff_weighted = ff * w_f

        # ↓ Step 7: Concat + restore (Fig.4 right)
        fused = torch.cat([fe_weighted, ff_weighted], dim=1)  # [B, 2C, H, W]
        out = self.restore_conv(fused)  # [B, C, H, W]
        return out


# ======== ✅ Updated SNNCNN3 with ASFM (CMM → ASFM → ConvLSTM) ========
class SNNCNN3(nn.Module):
    def __init__(self, num_classes=7, convlstm_hidden=256, use_hdr=True, hdr_alpha=0.4):
        super(SNNCNN3, self).__init__()
        self.use_hdr = use_hdr

        # ✅ HDR Enhancer for RGB only
        if use_hdr:
            self.hdr_enhancer = HDREnhancer(alpha=hdr_alpha)
            for p in self.hdr_enhancer.parameters():
                p.requires_grad = False

        # Dual ResNet-18 backbones
        self.resnet_frame = resnet18(pretrained=True)
        self.resnet_event = resnet18(pretrained=True)
        self.resnet_frame.fc = nn.Identity()
        self.resnet_event.fc = nn.Identity()

        # Hooks to extract layer2/3/4
        self.feat_rgb = {}
        self.feat_evt = {}
        def hook(name, feats):
            def fn(m, inp, out): feats[name] = out
            return fn

        self.resnet_frame.layer2.register_forward_hook(hook('layer2', self.feat_rgb))
        self.resnet_frame.layer3.register_forward_hook(hook('layer3', self.feat_rgb))
        self.resnet_frame.layer4.register_forward_hook(hook('layer4', self.feat_rgb))
        self.resnet_event.layer2.register_forward_hook(hook('layer2', self.feat_evt))
        self.resnet_event.layer3.register_forward_hook(hook('layer3', self.feat_evt))
        self.resnet_event.layer4.register_forward_hook(hook('layer4', self.feat_evt))

        # MFCM modules
        self.mfcm_frame = MFCM(in_channels=[128, 256, 512], align_dim=128, num_heads=4, k1=2, k2=3)
        self.mfcm_event = MFCM(in_channels=[128, 256, 512], align_dim=128, num_heads=4, k1=2, k2=3)

        # CMM
        self.cmm = CMMBlock(in_channels=512)

        # ✅ NEW: ASFM after CMM (replaces simple concat)
        self.asfm = ASFM(in_channels=512, reduction=16, alpha_amp=1.0, alpha_low=0.0)

        # ⚠️ ConvLSTM input_channels = 512 (was 1024)
        self.convlstm = ConvLSTM(
            input_channels=512,
            hidden_channels=convlstm_hidden,
            kernel_size=3,
            num_layers=1
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(convlstm_hidden, num_classes)

    def forward(self, frame, event):
        # Auto-adapt: [B, 3, H, W] → [B, 3, 1, H, W]
        if frame.dim() == 4:
            frame = frame.unsqueeze(2)
            event = event.unsqueeze(2)
        B, _, T, H, W = frame.shape
        assert frame.shape[1] == 3, f"Expected 3 channels, got {frame.shape}"
        assert event.shape == frame.shape, f"frame/event shape mismatch: {frame.shape} vs {event.shape}"

        # ✅ HDR Enhancement on RGB (only if enabled)
        if self.use_hdr:
            with torch.no_grad():
                frame = self.hdr_enhancer(frame)  # [B, 3, T, H, W]

        fused_seq_list = []
        for t in range(T):
            # RGB branch (now enhanced!)
            _ = self.resnet_frame(frame[:, :, t])
            f2_rgb = self.feat_rgb['layer2']
            f3_rgb = self.feat_rgb['layer3']
            f4_rgb = self.feat_rgb['layer4']
            F_frame_t = self.mfcm_frame([f2_rgb, f3_rgb, f4_rgb])  # [B, 512, H', W']

            # Event branch (unchanged)
            _ = self.resnet_event(event[:, :, t])
            f2_evt = self.feat_evt['layer2']
            f3_evt = self.feat_evt['layer3']
            f4_evt = self.feat_evt['layer4']
            F_event_t = self.mfcm_event([f2_evt, f3_evt, f4_evt])  # [B, 512, H', W']

            # CMM enhancement
            F_frame_enh_t, F_event_enh_t = self.cmm(F_frame_t, F_event_t)  # [B, 512, H', W'] ×2

            # ✅ ASFM adaptive fusion (replaces cat([F_frame_enh_t, F_event_enh_t]))
            fused_t = self.asfm(F_event_enh_t, F_frame_enh_t)  # [B, 512, H', W']

            fused_seq_list.append(fused_t)

        # → [B, T, 512, H', W']
        fused_seq = torch.stack(fused_seq_list, dim=1)

        # ConvLSTM
        h_last = self.convlstm(fused_seq)  # [B, 256, H', W']

        # Pooling → classification
        g = self.avgpool(h_last).flatten(1)
        logits = self.classifier(g)
        return logits


# ======== Utilities ========
def generate_model_snn(num_classes=7, convlstm_hidden=256, use_hdr=True, hdr_alpha=0.4):
    model = SNNCNN3(
        num_classes=num_classes,
        convlstm_hidden=convlstm_hidden,
        use_hdr=use_hdr,
        hdr_alpha=hdr_alpha
    )
    return model


def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model = model.to(device)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
        else:
            model = model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    return model