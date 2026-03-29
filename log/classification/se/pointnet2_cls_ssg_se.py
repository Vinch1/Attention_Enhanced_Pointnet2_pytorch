import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Adaptively recalibrates channel-wise feature responses by
    modeling channel interdependencies.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: [B, C, N]
        B, C, N = x.shape
        # Squeeze: global average pooling
        squeeze = x.mean(dim=2)  # [B, C]
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))  # [B, C]
        # Scale
        return x * excitation.unsqueeze(2)


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # Set Abstraction layers (same as pointnet2_cls_ssg)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # SE blocks after SA1 and SA2
        self.se1 = SEBlock(128, reduction=16)
        self.se2 = SEBlock(256, reduction=16)

        # Deeper classifier with pre-activation pattern: BN -> ReLU -> FC
        # 1024 -> 512 -> 256 -> 128 -> num_class
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.drop1 = nn.Dropout(0.3)

        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.3)

        self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.drop3 = nn.Dropout(0.2)

        self.bn4 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, num_class)

        # Residual projection layer (for future residual connections)
        self.res_proj = nn.Linear(1024, 128)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # Set Abstraction with SE attention
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l1_points = self.se1(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.se2(l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Flatten
        x = l3_points.view(B, 1024)

        # Pre-activation classifier: BN -> ReLU -> FC -> Dropout
        x = self.drop1(self.fc1(F.relu(self.bn1(x))))
        x = self.drop2(self.fc2(F.relu(self.bn2(x))))
        x = self.drop3(self.fc3(F.relu(self.bn3(x))))
        x = self.fc4(F.relu(self.bn4(x)))

        x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)
        return total_loss
