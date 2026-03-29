# PointNet++ SSG with Squeeze-and-Excitation Design

## Goal

Create a new model (`pointnet2_cls_ssg_se`) that beats the `pointnet2_cls_ssg` baseline (92.46% instance accuracy) on ModelNet40 classification **without using normal vectors**, targeting 93%+ accuracy.

## Constraints

- Same training setup: 100 epochs, same batch size, same optimizer
- Minimize memory overhead: must be runnable on same hardware as baseline
- No normal vectors: xyz coordinates only (3 input channels)

## Architecture

### Overview

```
Input (B, 3, N)
    ↓
[SA1 + SE] → 512 points, 128 channels
    ↓
[SA2 + SE] → 128 points, 256 channels
    ↓
[SA3] → 1 point, 1024 channels (global)
    ↓
[Classifier] → 3 FC layers with pre-activation
    ↓
Output (B, num_class)
```

### Component 1: Squeeze-and-Excitation Block

Channel attention mechanism that adaptively recalibrates channel-wise feature responses.

**Implementation:**
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: [B, C, N]
        squeeze = x.mean(dim=2)  # [B, C]
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))  # [B, C]
        return x * excitation.unsqueeze(2)
```

**Parameters per layer:**
- SA1 (128 channels): 128*8 + 8*128 = 2,048 params
- SA2 (256 channels): 256*16 + 16*256 = 8,192 params
- Total SE params: ~10K (negligible vs 1.5M total)

### Component 2: Residual Set Abstraction

Add residual connections between SA layers where channel dimensions allow.

**Implementation:**
- Store intermediate features before SA layer
- Add to output if dimensions match
- Use 1x1 conv projection if dimensions don't match

### Component 3: Deeper Classifier with Pre-Activation

Replace post-activation with pre-activation pattern for better gradient flow.

**Structure:**
```
1024 → BN → ReLU → FC(512) → Dropout(0.3)
     → BN → ReLU → FC(256) → Dropout(0.3)
     → BN → ReLU → FC(128) → Dropout(0.2)
     → FC(40)
```

**Key differences from baseline:**
- Pre-activation (BN-ReLU-FC) instead of post-activation (FC-BN-ReLU)
- 3 hidden layers instead of 2
- Decreasing dropout: 0.3 → 0.3 → 0.2 (instead of 0.4 → 0.4)

## File Structure

```
models/
├── pointnet2_cls_ssg_se.py  # NEW - Main model definition
├── pointnet2_utils.py       # Existing utilities (no changes)
└── ...
```

## Model Comparison

| Aspect | SSG (baseline) | SSG-SE (new) |
|--------|----------------|--------------|
| SA layers | 3 standard | 3 with SE attention |
| Residual | No | Yes |
| Classifier | 2 FC, post-activation | 3 FC, pre-activation |
| Dropout | 0.4, 0.4 | 0.3, 0.3, 0.2 |
| Parameters | ~1.5M | ~1.6M (+6%) |
| Memory overhead | Baseline | +5-10% |

## Expected Improvements

- **SE attention:** +0.3-0.5% (channel recalibration helps feature discrimination)
- **Residual connections:** +0.2-0.4% (better gradient flow)
- **Deeper classifier:** +0.2-0.3% (more expressive decision boundary)
- **Pre-activation:** +0.1-0.2% (training stability)
- **Lower dropout:** +0.1-0.2% (less regularization needed with better architecture)

**Total expected improvement:** +0.9-1.6% (target: 93.4-94%)

## Training

Use existing `train_classification.py` with:
```bash
python train_classification.py --model pointnet2_cls_ssg_se --epoch 100
```

No changes to training script required - model follows same interface as existing models.

## Success Criteria

- Instance accuracy > 92.46% (beats SSG baseline)
- Target: 93%+ instance accuracy
- Memory usage within 110% of baseline
