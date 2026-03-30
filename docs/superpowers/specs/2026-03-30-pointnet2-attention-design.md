# PointNet2 混合注意力模型设计

## 目标
在 PointNet2 SSG 分类模型基础上引入混合注意力机制，提升 ModelNet40 等基准上的分类精度。

## 整体架构

```
Input [B, 3/6, N]
       ↓
    SA1 (512 points) → Local Attn Pool → SE(128) → l1_points [B, 128, 512]
       ↓
    SA2 (128 points) → Local Attn Pool → SE(256) → l2_points [B, 256, 128]
       ↓
    SA3 (1 point, global) → l3_points [B, 1024, 1]
       ↓
    Cross-Level Attention (融合 l1, l2, l3) → [B, 256]
       ↓
    Classifier: FC(128) → FC(num_class)
       ↓
    Output [B, num_class]
```

## 核心组件

### 1. LocalAttentionPooling
替代 max pooling，在局部点群内做 attention 加权聚合：
- 输入: [B, C, nsample, npoint]
- 通过小型 MLP 计算每个点的 attention score
- 输出: [B, C, npoint]

### 2. SEBlock (保留现有实现)
通道注意力，自适应校准通道权重：
- Squeeze: 全局平均池化
- Excitation: FC → ReLU → FC → Sigmoid
- Scale: 原特征 × attention 权重

### 3. CrossLevelAttention
融合 SA1、SA2、SA3 三层特征：
- 各层特征投影到统一维度 (256)
- 全局池化后组成 3 个 token
- Multi-head self-attention (4 heads)
- LayerNorm + 残差连接

## 分类器
- 256 → 128 → num_class
- Pre-activation: BN → ReLU → FC → Dropout
- Dropout rate: 0.3

## 文件结构

```
models/
├── pointnet2_cls_ssg.py          # 原版
├── pointnet2_cls_ssg_se.py       # SE 版本
├── pointnet2_cls_ssg_attn.py     # 新增: 混合注意力版本
└── pointnet2_utils.py            # 工具函数
```

## 使用方式

```bash
python train_classification.py --model pointnet2_cls_ssg_attn
```

## 预期改进
- 更好的多尺度特征融合
- 捕获长距离依赖
- 参数量增加约 15-20%
