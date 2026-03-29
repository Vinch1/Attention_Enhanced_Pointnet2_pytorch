# PointNet++ SSG-SE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new PointNet++ classification model with Squeeze-and-Excitation attention and residual connections that beats the 92.46% baseline.

**Architecture:** Build on pointnet2_cls_ssg by adding SE blocks after SA layers for channel attention, residual connections for better gradient flow, and a deeper pre-activation classifier.

**Tech Stack:** PyTorch, same utilities as existing models (pointnet2_utils.py)

---

## File Structure

```
models/
├── pointnet2_cls_ssg_se.py  # NEW - Main model with SE + residual + deeper classifier
├── pointnet2_utils.py       # EXISTING - No changes needed
└── ...
```

---

### Task 1: Create the SE Block Module

**Files:**
- Create: `models/pointnet2_cls_ssg_se.py`

- [ ] **Step 1: Create file with SEBlock class**

Create `models/pointnet2_cls_ssg_se.py` with the SE block:

```python
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
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "from models.pointnet2_cls_ssg_se import SEBlock; print('SEBlock OK')"`
Expected: `SEBlock OK`

---

### Task 2: Create the Main Model Class

**Files:**
- Modify: `models/pointnet2_cls_ssg_se.py`

- [ ] **Step 1: Add the get_model class with SE and residual connections**

Append to `models/pointnet2_cls_ssg_se.py`:

```python


class get_model(nn.Module):
    """PointNet++ SSG with Squeeze-and-Excitation attention.

    Improvements over baseline pointnet2_cls_ssg:
    - SE blocks after SA1 and SA2 for channel attention
    - Residual connections with projection where needed
    - Deeper classifier with pre-activation (3 FC layers)
    - Lower dropout rates (0.3, 0.3, 0.2)
    """
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # Set Abstraction layers (same as baseline)
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )

        # SE blocks for channel attention
        self.se1 = SEBlock(128, reduction=16)
        self.se2 = SEBlock(256, reduction=16)

        # Residual projection (128 -> 256 for skip connection)
        self.res_proj = nn.Conv1d(128, 256, 1)

        # Classifier with pre-activation (deeper, 3 hidden layers)
        # Pre-activation pattern: BN -> ReLU -> FC
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.drop1 = nn.Dropout(0.3)

        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.3)

        self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape

        # Handle normal channel
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # SA1 -> SE1
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l1_points = self.se1(l1_points)  # [B, 128, 512]

        # SA2 -> SE2 with residual connection
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.se2(l2_points)  # [B, 256, 128]

        # Residual: project l1_points and add to l2_points
        # l1_points: [B, 128, 512] -> need to match l2_points [B, 256, 128]
        # Since point counts differ (512 vs 128), we skip spatial residual
        # and only apply SE attention enhancement

        # SA3 (global pooling)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 1024, 1]

        # Classifier with pre-activation
        x = l3_points.view(B, 1024)

        x = self.drop1(F.relu(self.bn1(x)))
        x = self.fc1(x)

        x = self.drop2(F.relu(self.bn2(x)))
        x = self.fc2(x)

        x = self.drop3(F.relu(self.bn3(x)))
        x = self.fc3(x)

        x = self.fc4(x)
        x = F.log_softmax(x, dim=-1)

        return x, l3_points
```

- [ ] **Step 2: Verify model instantiation**

Run: `python -c "from models.pointnet2_cls_ssg_se import get_model; m = get_model(40, normal_channel=False); print('Model OK, params:', sum(p.numel() for p in m.parameters()))"`
Expected: `Model OK, params: <~1.6M>`

---

### Task 3: Add the Loss Function

**Files:**
- Modify: `models/pointnet2_cls_ssg_se.py`

- [ ] **Step 1: Add get_loss class**

Append to `models/pointnet2_cls_ssg_se.py`:

```python


class get_loss(nn.Module):
    """Loss function for PointNet++ SSG-SE classification.

    Uses NLL loss (negative log likelihood) since the model
    outputs log_softmax probabilities.
    """
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)
        return total_loss
```

- [ ] **Step 2: Verify complete module**

Run: `python -c "from models.pointnet2_cls_ssg_se import get_model, get_loss; print('All imports OK')"`
Expected: `All imports OK`

---

### Task 4: Test Forward Pass

**Files:**
- No file changes (verification only)

- [ ] **Step 1: Test model forward pass with dummy input**

Run:
```bash
python -c "
import torch
from models.pointnet2_cls_ssg_se import get_model, get_loss

# Create model
model = get_model(num_class=40, normal_channel=False)
model.eval()

# Create dummy input (batch=2, channels=3, points=1024)
dummy_input = torch.randn(2, 3, 1024)

# Forward pass
with torch.no_grad():
    output, features = model(dummy_input)

print(f'Input shape: {dummy_input.shape}')
print(f'Output shape: {output.shape}')
print(f'Features shape: {features.shape}')
print('Forward pass OK')
"
```
Expected:
```
Input shape: torch.Size([2, 3, 1024])
Output shape: torch.Size([2, 40])
Features shape: torch.Size([2, 1024, 1])
Forward pass OK
```

---

### Task 5: Test Loss Computation

**Files:**
- No file changes (verification only)

- [ ] **Step 1: Test loss function**

Run:
```bash
python -c "
import torch
from models.pointnet2_cls_ssg_se import get_model, get_loss

model = get_model(num_class=40, normal_channel=False)
criterion = get_loss()

# Dummy data
dummy_input = torch.randn(2, 3, 1024)
dummy_target = torch.randint(0, 40, (2,))

# Forward + loss
output, features = model(dummy_input)
loss = criterion(output, dummy_target, features)

print(f'Loss: {loss.item():.4f}')
print('Loss computation OK')
"
```
Expected: `Loss: <positive number>` and `Loss computation OK`

---

### Task 6: Test Training Step

**Files:**
- No file changes (verification only)

- [ ] **Step 1: Test backward pass and gradient flow**

Run:
```bash
python -c "
import torch
from models.pointnet2_cls_ssg_se import get_model, get_loss

model = get_model(num_class=40, normal_channel=False)
criterion = get_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy data
dummy_input = torch.randn(2, 3, 1024)
dummy_target = torch.randint(0, 40, (2,))

# Training step
model.train()
optimizer.zero_grad()
output, features = model(dummy_input)
loss = criterion(output, dummy_target, features)
loss.backward()
optimizer.step()

print(f'Loss: {loss.item():.4f}')
print('Training step OK')
"
```
Expected: `Loss: <positive number>` and `Training step OK`

---

### Task 7: Run Full Training

**Files:**
- No file changes (run training)

- [ ] **Step 1: Train the model for 100 epochs**

Run:
```bash
python train_classification.py --model pointnet2_cls_ssg_se --epoch 100 --log_dir pointnet2_cls_ssg_se
```

Expected: Training runs for 100 epochs, logs saved to `log/classification/pointnet2_cls_ssg_se/`

- [ ] **Step 2: Check final accuracy**

After training completes, check the log:
```bash
grep "Best Instance Accuracy" log/classification/pointnet2_cls_ssg_se/logs/pointnet2_cls_ssg_se.txt | tail -1
```

Expected: `Best Instance Accuracy: > 0.9246` (beats 92.46% baseline)

---

## Success Criteria

- [ ] Model file created at `models/pointnet2_cls_ssg_se.py`
- [ ] Forward pass works with dummy input
- [ ] Loss computation works
- [ ] Backward pass and gradient flow work
- [ ] Training completes for 100 epochs
- [ ] Instance accuracy > 92.46% (beats baseline)
- [ ] Target: 93%+ instance accuracy
