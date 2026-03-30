import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Ensure reduction doesn't make intermediate dim too small
        reduction = min(reduction, channels // 4)
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)

    def forward(self, x):
        # x: [B, C, N]
        B, C, N = x.shape
        squeeze = x.mean(dim=2)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        return x * excitation.unsqueeze(2)


class LocalAttentionPooling(nn.Module):
    """Attention-based pooling for local point groups.

    Replaces max pooling with learned attention weights.
    Falls back to max pooling if nsample <= 1.
    """
    def __init__(self, in_channels):
        super(LocalAttentionPooling, self).__init__()
        hidden = max(1, in_channels // 4)
        self.score_fn = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, x):
        # x: [B, C, nsample, npoint]
        B, C, nsample, npoint = x.shape

        # Fallback to max pooling for edge cases
        if nsample <= 1:
            return x.squeeze(2)

        scores = self.score_fn(x)  # [B, 1, nsample, npoint]
        weights = F.softmax(scores, dim=2)
        return (x * weights).sum(dim=2)  # [B, C, npoint]


class CrossLevelAttention(nn.Module):
    """Cross-level attention to fuse features from multiple SA layers."""
    def __init__(self, channels_list=(128, 256, 1024), hidden_dim=256, num_heads=4):
        super(CrossLevelAttention, self).__init__()
        self.proj_l1 = nn.Linear(channels_list[0], hidden_dim)
        self.proj_l2 = nn.Linear(channels_list[1], hidden_dim)
        self.proj_l3 = nn.Linear(channels_list[2], hidden_dim)

        self.mha = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, l1_points, l2_points, l3_points):
        # l1: [B, 128, 512], l2: [B, 256, 128], l3: [B, 1024, 1]
        # Global pooling for each level
        f1 = self.proj_l1(l1_points.mean(dim=-1))  # [B, hidden_dim]
        f2 = self.proj_l2(l2_points.mean(dim=-1))  # [B, hidden_dim]
        f3 = self.proj_l3(l3_points.squeeze(-1))   # [B, hidden_dim]

        tokens = torch.stack([f1, f2, f3], dim=1)  # [B, 3, hidden_dim]
        tokens = tokens.contiguous()
        attn_out, _ = self.mha(tokens, tokens, tokens)
        out = self.norm(tokens + attn_out)
        out = F.relu(self.fc_out(out))
        return out.mean(dim=1)  # [B, hidden_dim]


class PointNetSetAbstractionAttn(nn.Module):
    """Set Abstraction with Local Attention Pooling."""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstractionAttn, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # Local attention pooling instead of max pooling
        self.attn_pool = LocalAttentionPooling(mlp[-1])

    def forward(self, xyz, points):
        from pointnet2_utils import sample_and_group, sample_and_group_all

        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_points: [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()  # [B, C+D, nsample, npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Use attention pooling instead of max pooling
        new_points = self.attn_pool(new_points)  # [B, C, npoint]
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()

        return new_xyz, new_points


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # Set Abstraction with Attention Pooling
        self.sa1 = PointNetSetAbstractionAttn(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstractionAttn(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstractionAttn(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )

        # SE blocks for channel attention
        self.se1 = SEBlock(128, reduction=16)
        self.se2 = SEBlock(256, reduction=16)

        # Cross-level attention
        self.cross_attn = CrossLevelAttention(
            channels_list=(128, 256, 1024),
            hidden_dim=256,
            num_heads=4
        )

        # Classifier
        self.bn1 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :].contiguous()
            xyz = xyz[:, :3, :].contiguous()
        else:
            norm = None

        # Set Abstraction with local attention + SE
        l1_xyz, l1_points = self.sa1(xyz.contiguous(), norm)
        l1_points = self.se1(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz.contiguous(), l1_points.contiguous())
        l2_points = self.se2(l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz.contiguous(), l2_points.contiguous())

        # Cross-level attention fusion
        x = self.cross_attn(l1_points, l2_points, l3_points)  # [B, 256]

        # Classifier
        x = self.drop1(F.relu(self.bn1(x)))
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)
        return total_loss
