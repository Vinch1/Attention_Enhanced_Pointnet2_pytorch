import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import (
    farthest_point_sample,
    index_points,
    sample_and_group_all,
    square_distance,
)


def _is_mps_tensor(x):
    return x.device.type == 'mps'


def _masked_softmax(scores, mask, dim):
    """Softmax that stays stable on MPS and ignores padded neighbors."""
    if mask is None:
        return F.softmax(scores, dim=dim)

    mask = mask.to(dtype=torch.bool, device=scores.device)
    work_scores = scores.float() if _is_mps_tensor(scores) and scores.dtype != torch.float32 else scores

    # Very negative finite values are safer than -inf on MPS.
    fill_value = -1e4 if work_scores.dtype in (torch.float16, torch.bfloat16) else -1e9
    filled_scores = torch.where(mask, work_scores, torch.full_like(work_scores, fill_value))
    weights = F.softmax(filled_scores, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))

    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    weights = weights / denom
    return weights.to(dtype=scores.dtype)


def query_ball_point_with_mask(radius, nsample, xyz, new_xyz):
    """
    Radius query with an explicit validity mask.

    The original implementation duplicates the first valid neighbor to fill
    padded slots. That is harmless for max pooling but biases attention
    pooling, so this function returns both the gathered indices and a mask.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    all_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).expand(B, S, N)
    sqrdists = square_distance(new_xyz, xyz)
    valid = sqrdists <= (radius ** 2)

    masked_idx = torch.where(valid, all_idx, torch.full_like(all_idx, N))
    group_idx = masked_idx.sort(dim=-1)[0][:, :, :nsample]
    valid_mask = group_idx != N

    nearest_idx = sqrdists.argmin(dim=-1, keepdim=True).expand(-1, -1, nsample)
    first_valid = group_idx[:, :, :1].expand(-1, -1, nsample)
    safe_fill = torch.where(valid_mask[:, :, :1].expand(-1, -1, nsample), first_valid, nearest_idx)
    group_idx = torch.where(valid_mask, group_idx, safe_fill)
    return group_idx, valid_mask


def sample_and_group_with_mask(npoint, radius, nsample, xyz, points):
    """PointNet++ grouping with an explicit neighbor-validity mask."""
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx, valid_mask = query_ball_point_with_mask(radius, nsample, xyz, new_xyz)

    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points, valid_mask


def sample_and_group_all_with_mask(xyz, points):
    new_xyz, new_points = sample_and_group_all(xyz, points)
    B, _, N, _ = new_points.shape
    valid_mask = torch.ones((B, 1, N), dtype=torch.bool, device=xyz.device)
    return new_xyz, new_points, valid_mask


class SEBlock(nn.Module):
    """Channel recalibration for [B, C, N] features."""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        hidden = max(1, channels // max(1, reduction))
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        squeeze = x.mean(dim=2)
        excitation = F.relu(self.fc1(squeeze), inplace=True)
        excitation = torch.sigmoid(self.fc2(excitation))
        return x * excitation.unsqueeze(2)


class HybridLocalPooling(nn.Module):
    """Masked local attention pooling with a max-pooling residual path."""

    def __init__(self, in_channels):
        super(HybridLocalPooling, self).__init__()
        hidden = max(16, in_channels // 4)
        self.score_fn = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        self.value_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, valid_mask=None):
        # x: [B, C, nsample, npoint]
        if x.shape[2] <= 1:
            return x.squeeze(2)

        attn_scores = self.score_fn(x).squeeze(1)
        if valid_mask is not None:
            attn_weights = _masked_softmax(attn_scores, valid_mask, dim=1)
        else:
            attn_weights = F.softmax(attn_scores, dim=1)

        value = self.value_proj(x)
        attn_pooled = (value * attn_weights.unsqueeze(1)).sum(dim=2)
        max_pooled = torch.max(x, dim=2)[0]
        return self.fuse(torch.cat([attn_pooled, max_pooled], dim=1))


class PointNetSetAbstractionAttnV2(nn.Module):
    """Set abstraction with masked attention pooling."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstractionAttnV2, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.pool = HybridLocalPooling(mlp[-1])

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()

        if self.group_all:
            new_xyz, new_points, valid_mask = sample_and_group_all_with_mask(xyz, points)
        else:
            new_xyz, new_points, valid_mask = sample_and_group_with_mask(
                self.npoint, self.radius, self.nsample, xyz, points
            )

        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)), inplace=True)

        pooled_mask = valid_mask.permute(0, 2, 1).contiguous()
        new_points = self.pool(new_points, pooled_mask)
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class CrossLevelAttentionV2(nn.Module):
    """CLS-token fusion over multi-scale summaries plus a global residual."""

    def __init__(self, channels_list=(128, 256, 1024), hidden_dim=256, num_heads=4, dropout=0.1):
        super(CrossLevelAttentionV2, self).__init__()
        self.proj_l1 = nn.Linear(channels_list[0], hidden_dim)
        self.proj_l2 = nn.Linear(channels_list[1], hidden_dim)
        self.proj_l3 = nn.Linear(channels_list[2], hidden_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.global_residual = nn.Sequential(
            nn.Linear(channels_list[2], hidden_dim),
            nn.ReLU(inplace=True),
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, l1_points, l2_points, l3_points):
        l1_mean = l1_points.mean(dim=-1)
        l1_max = l1_points.max(dim=-1)[0]
        l2_mean = l2_points.mean(dim=-1)
        l2_max = l2_points.max(dim=-1)[0]
        l3_global = l3_points.squeeze(-1)

        tokens = torch.stack(
            [
                self.proj_l1(l1_mean),
                self.proj_l1(l1_max),
                self.proj_l2(l2_mean),
                self.proj_l2(l2_max),
                self.proj_l3(l3_global),
            ],
            dim=1,
        )
        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1).contiguous()

        attn_input = self.input_norm(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        attn_out = torch.nan_to_num(attn_out)
        tokens = self.norm1(tokens + attn_out)

        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        cls_out = tokens[:, 0]
        global_out = self.global_residual(l3_global)
        return torch.cat([cls_out, global_out], dim=-1)


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstractionAttnV2(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=in_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstractionAttnV2(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstractionAttnV2(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        self.se1 = SEBlock(128, reduction=16)
        self.se2 = SEBlock(256, reduction=16)
        self.se3 = SEBlock(1024, reduction=16)

        self.cross_attn = CrossLevelAttentionV2(
            channels_list=(128, 256, 1024),
            hidden_dim=256,
            num_heads=4,
            dropout=0.1,
        )

        self.bn1 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        self.drop1 = nn.Dropout(0.4)

        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)

        self.bn3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, xyz):
        if self.normal_channel:
            norm = xyz[:, 3:, :].contiguous()
            xyz = xyz[:, :3, :].contiguous()
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz.contiguous(), norm)
        l1_points = self.se1(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz.contiguous(), l1_points.contiguous())
        l2_points = self.se2(l2_points)

        _, l3_points = self.sa3(l2_xyz.contiguous(), l2_points.contiguous())
        l3_points = self.se3(l3_points)

        x = self.cross_attn(l1_points, l2_points, l3_points)
        x = self.drop1(self.fc1(F.relu(self.bn1(x), inplace=True)))
        x = self.drop2(self.fc2(F.relu(self.bn2(x), inplace=True)))
        x = self.fc3(F.relu(self.bn3(x), inplace=True))
        x = F.log_softmax(x, dim=-1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        return F.nll_loss(pred, target)
