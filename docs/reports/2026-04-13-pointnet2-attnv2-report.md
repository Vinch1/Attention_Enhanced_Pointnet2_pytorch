# PointNet++ Attention Enhancement Project Report

## 1. Purpose of This Report

This report documents the full project trajectory, including:

- what was changed in the codebase
- why each architectural or engineering change was made
- the underlying principles behind each change
- all available experimental metrics and results
- how the final method compares with the corresponding PointNet++ paper
- how the method behaves on a second dataset
- the current drawbacks and limitations

This document is written to be reused directly in a future session for generating a PowerPoint presentation.

The intended PPT topics are:

- Introduction of the project
- Compare experimental results with the corresponding paper
- Analyze the performance on different datasets
- Analyze the drawbacks of the method


## 2. Executive Summary

The project started from the upstream PointNet++ SSG classification baseline in `models/pointnet2_cls_ssg.py`.

The work then progressed in three main stages:

- Apple Silicon / MPS compatibility and training stability fixes
- architecture exploration on ModelNet40: `SE`, `attn-v1`, and `attn-v2`
- dataset extension and experiment logging on `3D MNIST`

The main experimental conclusions are:

- The local PointNet++ SSG baseline on ModelNet40 is already strong: **92.459% instance accuracy / 89.175% class accuracy**
- `SE` improves class accuracy but does not beat the baseline on instance accuracy
- `attn-v1` underperforms clearly because of design flaws in local attention pooling and the classifier head
- `attn-v2` fixes the main problems of `attn-v1`, recovers most of the lost performance, and improves strongly over `attn-v1`, but it still does **not** beat the local PointNet++ SSG baseline on ModelNet40 instance accuracy
- On `3D MNIST`, `attn-v2` slightly beats the baseline under the same training setup, suggesting the revised attention design is not fundamentally broken, but the gain is small because the dataset is relatively easy

The most defensible final conclusion is:

> The `attn-v2` method is a meaningful correction over `attn-v1` and shows slightly better cross-dataset behavior on `3D MNIST`, but on ModelNet40 it does not surpass a strong local PointNet++ SSG baseline. The method improves class balance more consistently than top-line instance accuracy, and its parameter cost is significantly higher than the baseline.


## 3. Branch and Engineering Timeline

### 3.1 Branch Overview

The repository currently contains these main branches:

| Branch | Role |
|---|---|
| `master` | upstream baseline repository |
| `feat/apple_silicon` | Apple Silicon / MPS support, baseline reruns, early SE and attention experiments |
| `pointnet2-attn-v2-mac-safe` | `attn-v2` model design and ModelNet40 experiment |
| `3dmnist-attnv2-training` | `3D MNIST` dataloader, dataset-aware train/test pipeline, persistent logs, cross-dataset experiments |

### 3.2 Key Commit Trajectory

Important commits in the experimental evolution:

- `2284629`: refactor data loading and augmentations for Apple Silicon
- `64b5aef`: new model work and Apple Silicon fixes
- `3a536bf`: SE classification experiment result
- `1deb3e0` / `b52b628`: early attention model iterations
- `170cabf`: `attn-v2` introduced and validated against `attn-v1`
- `c51fe93`: `3D MNIST` integration, logging pipeline, baseline + attn-v2 comparison on second dataset

### 3.3 What Changed by Branch

Relative to `master`, the branch work can be summarized as follows.

#### `feat/apple_silicon`

Main engineering additions:

- `device_utils.py` added for automatic `MPS / CUDA / CPU` selection
- MPS-safe fixes in `models/pointnet2_utils.py`
- Darwin-aware worker defaults in training scripts
- `SE` model added: `models/pointnet2_cls_ssg_se.py`
- `attn-v1` model added: `models/pointnet2_cls_ssg_attn.py`
- multiple local experiment logs added under `log/classification/`

Main principle:

- make the original repository train reliably on Apple Silicon first
- then iterate on the classification architecture

#### `pointnet2-attn-v2-mac-safe`

Main additions:

- new model `models/pointnet2_cls_ssg_attn_v2.py`
- ModelNet40 experiment log in `log/classification/attnv2/`
- extra MPS-safe attention logic

Main principle:

- preserve the useful idea of attention-based aggregation
- remove the failure modes that made `attn-v1` unstable or ineffective

#### `3dmnist-attnv2-training`

Main additions:

- `data_utils/ThreeDMNISTDataLoader.py`
- dataset-aware `train_classification.py`
- dataset-aware `test_classification.py`
- `3D MNIST` baseline and `attn-v2` experiment folders
- permanent run artifacts: `run_config.json`, `metrics.csv`, `summary.json`
- `h5py` dependency added to `pyproject.toml`
- `.gitignore` updated to avoid versioning large `.h5` datasets and `.pth` checkpoints

Main principle:

- convert one-off experiments into reproducible and portable experimental records


## 4. Baseline Model and Optimization Path

### 4.1 Baseline PointNet++ SSG

Baseline file:

- `models/pointnet2_cls_ssg.py`

Structure:

- SA1: `512` points, `radius=0.2`, `nsample=32`, channels `64 -> 64 -> 128`
- SA2: `128` points, `radius=0.4`, `nsample=64`, channels `128 -> 128 -> 256`
- SA3: global set abstraction, channels `256 -> 512 -> 1024`
- classifier: `1024 -> 512 -> 256 -> num_class`

Relevant code:

- `models/pointnet2_cls_ssg.py:11`
- `models/pointnet2_cls_ssg.py:12`
- `models/pointnet2_cls_ssg.py:13`
- `models/pointnet2_cls_ssg.py:14`

Core principle:

- hierarchical local-to-global feature extraction using set abstraction


### 4.2 SE Version

SE model file:

- `models/pointnet2_cls_ssg_se.py`

Main changes:

- channel attention after `SA1` and `SA2`
- deeper classifier with `BN -> ReLU -> FC -> Dropout`

Relevant code:

- `models/pointnet2_cls_ssg_se.py:7`
- `models/pointnet2_cls_ssg_se.py:41`
- `models/pointnet2_cls_ssg_se.py:45`

Reasoning:

- SE blocks recalibrate feature channels and may suppress noisy channels
- a deeper classifier should improve the expressive power of the decision boundary

Principle:

- channel-wise attention
- better gradient flow through pre-activation style classifier layers


### 4.3 Attention v1

Attention-v1 model file:

- `models/pointnet2_cls_ssg_attn.py`

Main changes:

- `LocalAttentionPooling` replaces max pooling inside SA layers
- `SE` remains on intermediate levels
- `CrossLevelAttention` fuses `l1`, `l2`, `l3`

Relevant code:

- `models/pointnet2_cls_ssg_attn.py:25`
- `models/pointnet2_cls_ssg_attn.py:54`
- `models/pointnet2_cls_ssg_attn.py:81`
- `models/pointnet2_cls_ssg_attn.py:151`

Original design intent:

- better local aggregation than max pooling
- better multi-scale fusion than using only the final global feature

But two major problems were identified.

Problem 1:

- `query_ball_point` in `models/pointnet2_utils.py:90` fills missing neighbors by duplicating the first valid point
- this behavior is safe for max pooling
- it is not safe for softmax attention because repeated padding points distort the weight distribution

Problem 2:

- the classifier head is effectively too shallow
- the code at `models/pointnet2_cls_ssg_attn.py:185` to `models/pointnet2_cls_ssg_attn.py:187` performs `BN/ReLU/Dropout`, then two linear layers with no real hidden nonlinearity between them

Additional weakness:

- `CrossLevelAttention` compresses each hierarchy level into a single mean-pooled token, then averages the output tokens again
- this collapses much of the spatial structure before the attention module can use it


### 4.4 Attention v2

Final optimized model file:

- `models/pointnet2_cls_ssg_attn_v2.py`

This was designed as a correction of `attn-v1`, not just an extension.

#### 4.4.1 Masked local attention

Relevant code:

- `models/pointnet2_cls_ssg_attn_v2.py:17`
- `models/pointnet2_cls_ssg_attn_v2.py:36`
- `models/pointnet2_cls_ssg_attn_v2.py:63`

What changed:

- explicit validity masks are produced during radius grouping
- padded neighbors are masked before softmax
- nearest valid neighbors are used only for safe indexing, not for fake attention mass

Why:

- prevent duplicated padding points from receiving real attention weight

Principle:

- attention should only be normalized over valid elements


#### 4.4.2 Hybrid local pooling

Relevant code:

- `models/pointnet2_cls_ssg_attn_v2.py:104`

What changed:

- attention pooling and max pooling are fused

Why:

- max pooling is very stable in PointNet / PointNet++
- attention pooling is expressive but more fragile
- combining them gives a residual-like safeguard

Principle:

- retain the robustness of max pooling while adding learnable weighting


#### 4.4.3 Stronger cross-level fusion

Relevant code:

- `models/pointnet2_cls_ssg_attn_v2.py:186`

What changed:

- use both `mean` and `max` summaries for `l1` and `l2`
- add a learnable `CLS` token
- preserve a direct residual path from the strong global feature `l3`
- use `LayerNorm + MHA + FFN`

Why:

- avoid collapsing all multi-scale information into only three mean-pooled vectors
- preserve the strongest global descriptor instead of fully averaging it away

Principle:

- token-based fusion should combine multi-scale statistics and preserve a global shortcut


#### 4.4.4 Stronger classifier

Relevant code:

- `models/pointnet2_cls_ssg_attn_v2.py:292`
- `models/pointnet2_cls_ssg_attn_v2.py:319`

What changed:

- classifier widened to `512 -> 256 -> 128 -> num_class`
- explicit nonlinear transitions restored

Why:

- fix the near-linear behavior of `attn-v1`

Principle:

- if the feature fusion stage is richer, the classifier must also have enough capacity to exploit it


#### 4.4.5 MPS / macOS protection

Relevant code:

- `models/pointnet2_utils.py:76`
- `models/pointnet2_utils.py:83`
- `models/pointnet2_utils.py:109`
- `models/pointnet2_cls_ssg_attn_v2.py:13`
- `models/pointnet2_cls_ssg_attn_v2.py:25`
- `models/pointnet2_cls_ssg_attn_v2.py:239`

What changed:

- clamp potentially unsafe indices
- replace boolean assignment patterns with `torch.where`
- avoid `-inf` masking in attention on MPS
- sanitize attention outputs with `torch.nan_to_num`
- enforce `contiguous()` after permutations

Why:

- Apple Silicon / MPS is workable but less forgiving than CUDA in some indexing and masking cases

Principle:

- numerical safety and index safety are first-class engineering requirements on MPS


## 5. Training and Evaluation Infrastructure Changes

### 5.1 Device and Runtime Support

Relevant file:

- `device_utils.py`

What was added:

- automatic device choice across `MPS`, `CUDA`, and `CPU`
- readable device naming for logs


### 5.2 3D MNIST Dataloader

Relevant file:

- `data_utils/ThreeDMNISTDataLoader.py`

Key behavior:

- reads `train_point_clouds.h5` and `test_point_clouds.h5`
- uses HDF5 group attributes for labels
- uses `points` or `points + normals` depending on `--use_normals`
- downsamples to `args.num_point`
- pads by resampling if a point cloud has fewer than `num_point` points
- normalizes XYZ with `pc_normalize`

Dataset characteristics observed locally:

- train samples: `5000`
- test samples: `1000`
- classes: `10`
- train point-count range: `9250` to `38200`
- test point-count range: `9700` to `34800`
- average train point count: `22600.09`
- average test point count: `21950.35`

Local class distribution:

- train labels: `{0: 479, 1: 563, 2: 488, 3: 493, 4: 535, 5: 434, 6: 501, 7: 550, 8: 462, 9: 495}`
- test labels: `{0: 85, 1: 126, 2: 116, 3: 107, 4: 110, 5: 87, 6: 87, 7: 99, 8: 89, 9: 94}`

Interpretation:

- class balance is reasonably good
- this makes class accuracy meaningful, not just instance accuracy


### 5.3 Persistent Logging

Relevant file:

- `train_classification.py`

Relevant code:

- dataset-aware config: `train_classification.py:109`
- artifact copying: `train_classification.py:173`
- persistent metadata: `train_classification.py:267`
- per-epoch metrics CSV: `train_classification.py:363`
- latest and best checkpoints: `train_classification.py:377`

What changed:

- `--dataset` and `--data_path` arguments added
- `run_config.json` saves the exact run setup
- `metrics.csv` saves per-epoch metrics
- `summary.json` saves final best metrics
- experiment directory copies the exact model and scripts used

Why:

- the earlier ModelNet40 logs were still usable, but they were mostly text-only
- the new setup makes each experiment self-contained and reproducible


### 5.4 Evaluation Script Upgrade

Relevant file:

- `test_classification.py`

What changed:

- `--dataset` and `--data_path` support
- automatic reuse of `run_config.json`

Why:

- evaluation should not depend on manually remembering the exact dataset settings from training


## 6. Evaluation Metrics

Two metrics are used throughout classification experiments.

### 6.1 Instance Accuracy

Definition:

- percentage of correctly classified samples over the whole test set

Interpretation:

- this is the main top-line benchmark metric in the original PointNet / PointNet++ literature


### 6.2 Class Accuracy

Definition:

- average of per-class accuracies

Interpretation:

- this measures class balance
- it is useful when one wants to know whether the model favors frequent or easy categories


### 6.3 Additional Logged Metrics

For the newer pipeline, these are also stored:

- train instance accuracy per epoch
- test instance accuracy per epoch
- test class accuracy per epoch
- best-so-far instance accuracy
- best-so-far class accuracy
- learning rate
- global step


## 7. Parameter Cost

Parameter counts were measured locally.

| Model | Parameters | Delta vs baseline |
|---|---:|---:|
| `pointnet2_cls_ssg` | 1,475,688 | baseline |
| `pointnet2_cls_ssg_se` | 1,647,616 | +11.65% |
| `pointnet2_cls_ssg_attn` | 1,831,683 | +24.12% |
| `pointnet2_cls_ssg_attn_v2` | 5,951,459 | +303.30% |

Important observation:

- `attn-v2` is much heavier than originally intended
- the largest parameter block is `sa3.pool.fuse.0.weight` in `models/pointnet2_cls_ssg_attn_v2.py`, which alone contains `2,097,152` parameters

Implication:

- the current `attn-v2` should be treated as an accuracy-oriented prototype, not an efficiency-optimized architecture


## 8. Experimental Setup

### 8.1 ModelNet40 Experiments

All ModelNet40 experiments summarized here used:

- `num_point = 1024`
- `use_normals = False`
- optimizer: `Adam`
- learning rate: `0.001`
- weight decay: `1e-4`
- scheduler: `StepLR(step_size=20, gamma=0.7)`
- device: Apple Silicon MPS

Important note:

- batch size was not identical across all ModelNet40 variants
- baseline and `attn-v1` used `batch_size=24`
- `SE` and `attn-v2` used `batch_size=16`

This means the ModelNet40 comparison is informative, but not perfectly controlled.


### 8.2 3D MNIST Experiments

Both `3D MNIST` comparison runs used the same fair setting:

- `num_point = 1024`
- `use_normals = False`
- `epoch = 100`
- `batch_size = 16`
- optimizer: `Adam`
- learning rate: `0.001`
- weight decay: `1e-4`
- device: Apple Silicon MPS
- effective dataloader workers: `0` for HDF5 safety

This is the cleanest head-to-head comparison in the project.


## 9. Experimental Results

### 9.1 ModelNet40 Results

#### 9.1.1 Main Comparison Table

| Model | Branch stage | Batch size | Best checkpoint epoch | Best instance acc | Best class acc | Final train inst acc | Final test inst acc | Final test class acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `pointnet2_cls_ssg` | Apple Silicon baseline rerun | 24 | 68 | 92.459% | 89.175% | 95.488% | 91.562% | 88.557% |
| `pointnet2_cls_ssg_se` | SE exploration | 16 | 92 | 91.976% | 89.626% | 92.927% | 91.048% | 87.578% |
| `pointnet2_cls_ssg_attn` | attention v1 | 24 | 88 | 91.392% | 86.231% | 92.480% | 89.798% | 84.697% |
| `pointnet2_cls_ssg_attn_v2` | attention v2 | 16 | 100 | 92.056% | 89.597% | 94.421% | 92.056% | 89.597% |

Notes:

- the baseline log contains an aborted 200-epoch start followed by the actual comparable 100-epoch run
- the best checkpoint epoch was inferred from the last `Saving at ...` entry in each text log
- for `attn-v2`, the best checkpoint was saved at epoch 100


#### 9.1.2 Interpretation

`SE`:

- instance accuracy dropped by `0.484` percentage points relative to baseline
- class accuracy increased by `0.450` percentage points relative to baseline

Meaning:

- channel recalibration appears to help class balance
- it did not help enough to surpass the already strong baseline on overall instance accuracy

`attn-v1`:

- instance accuracy dropped by `1.069` points relative to baseline
- class accuracy dropped by `2.945` points relative to baseline

Meaning:

- the initial attention design was clearly worse than the baseline
- the discovered implementation and architecture issues were materially harmful, not cosmetic

`attn-v2`:

- instance accuracy improved over `attn-v1` by `0.665` points
- class accuracy improved over `attn-v1` by `3.367` points
- instance accuracy still remained `0.403` points below the baseline
- class accuracy was `0.422` points above the baseline

Meaning:

- the `attn-v2` corrections worked
- the revised model no longer behaves like a failed attention prototype
- but on ModelNet40 it still does not justify replacing the baseline if instance accuracy is the main target


### 9.2 3D MNIST Results

#### 9.2.1 Main Comparison Table

| Model | Dataset | Epochs | Batch size | Best instance acc | Best class acc | Best instance epoch | Best class epoch | Final train inst acc | Final test inst acc | Final test class acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pointnet2_cls_ssg` | `3D MNIST` | 100 | 16 | 98.611% | 98.623% | 60 | 100 | 98.077% | 98.611% | 98.623% |
| `pointnet2_cls_ssg_attn_v2` | `3D MNIST` | 100 | 16 | 98.710% | 98.939% | 87 | 96 | 98.558% | 98.512% | 98.257% |

#### 9.2.2 Delta

`attn-v2` vs baseline on `3D MNIST`:

- best instance accuracy: `+0.099` percentage points
- best class accuracy: `+0.316` percentage points

Interpretation:

- the gain is small, but it is real
- the stronger gain is in class accuracy rather than pure instance accuracy
- this is consistent with the ModelNet40 pattern: the attention-enhanced models tend to help class balance more than top-line instance accuracy


#### 9.2.3 Convergence Behavior

Selected epoch snapshots show a useful difference between the baseline and `attn-v2`.

`attn-v2` on `3D MNIST`:

| Epoch | Train inst acc | Test inst acc | Test class acc |
|---|---:|---:|---:|
| 1 | 27.264% | 37.798% | 36.664% |
| 5 | 81.831% | 89.683% | 89.636% |
| 10 | 88.201% | 89.583% | 89.268% |
| 20 | 93.590% | 95.139% | 95.583% |
| 40 | 96.094% | 97.718% | 97.846% |
| 60 | 97.596% | 98.313% | 98.283% |
| 80 | 98.177% | 98.115% | 97.673% |
| 100 | 98.558% | 98.512% | 98.257% |

Baseline on `3D MNIST`:

| Epoch | Train inst acc | Test inst acc | Test class acc |
|---|---:|---:|---:|
| 1 | 51.542% | 71.726% | 70.406% |
| 5 | 85.377% | 87.599% | 87.867% |
| 10 | 89.744% | 83.631% | 83.431% |
| 20 | 92.869% | 94.246% | 94.517% |
| 40 | 95.192% | 96.627% | 96.702% |
| 60 | 96.334% | 98.611% | 98.433% |
| 80 | 97.135% | 98.016% | 97.997% |
| 100 | 98.077% | 98.611% | 98.623% |

Interpretation:

- the baseline learns much faster in the first epoch
- `attn-v2` starts slower, then gradually overtakes baseline in class-aware performance
- the baseline reaches its best instance accuracy earlier
- `attn-v2` continues improving class balance deeper into training


## 10. Comparison with the Corresponding Paper

### 10.1 Which Paper Is the Correct Comparison Target

The corresponding paper for the baseline method is:

- PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
- NeurIPS 2017
- Proceedings page: <https://proceedings.neurips.cc/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html>
- PDF: <https://proceedings.neurips.cc/paper_files/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf>

The paper reports ModelNet40 classification performance for the original PointNet++.

### 10.2 Paper Numbers

From the PointNet++ paper:

- PointNet++ without normals: **90.7%**
- PointNet++ with normals: **91.9%**

Important limitation:

- the paper reports overall classification accuracy, not the class accuracy metric used in our local logs
- therefore the direct paper comparison is only valid for instance accuracy


### 10.3 Comparison Table

| Method | Features | Accuracy source | Instance accuracy |
|---|---|---|---:|
| PointNet++ (paper) | XYZ only | official paper | 90.7% |
| PointNet++ (paper) | XYZ + normals | official paper | 91.9% |
| PointNet2_SSG (repo README) | XYZ only | repository README | 92.2% |
| PointNet2_SSG (repo README) | XYZ + normals | repository README | 92.4% |
| Our local PointNet2_SSG baseline | XYZ only | local retrain | 92.459% |
| Our local SE model | XYZ only | local retrain | 91.976% |
| Our local attention v1 | XYZ only | local retrain | 91.392% |
| Our local attention v2 | XYZ only | local retrain | 92.056% |


### 10.4 Paper Comparison Interpretation

Against the original PointNet++ paper result without normals:

- our local baseline is `+1.759` points higher
- our `attn-v2` is `+1.357` points higher

Against the paper result with normals:

- our local baseline is still `+0.559` points higher despite using only XYZ
- our `attn-v2` is `+0.156` points higher despite using only XYZ

However, the correct interpretation is not simply "our method beats the paper".

The more careful interpretation is:

- the modern PyTorch codebase already includes stronger training practices than the original 2017 setup
- the local baseline is already stronger than the official paper result
- therefore the real benchmark for the project is not the paper alone, but the **local baseline retrain**

This is why `attn-v2` should be judged mainly against the local PointNet++ SSG baseline, not only against the paper number.


## 11. Analysis on Different Datasets

### 11.1 Why the Second Dataset Was Needed

The project requirement asked for validation on another dataset to test performance and generality.

Using a second dataset answers a different question from the ModelNet40 experiment.

ModelNet40 answers:

- does the method work on a standard CAD-based benchmark?

`3D MNIST` answers:

- does the modified architecture still work when the task and data distribution change?


### 11.2 Why 3D MNIST Was Chosen

`3D MNIST` was available locally and had two practical advantages:

- it already contained point clouds in HDF5 format
- it was much easier to integrate than `ScanObjectNN` or raw `ShapeNetCore`

Why it is acceptable:

- it is a real second 3D classification dataset
- it allows controlled retraining and testing of both baseline and improved models

Why it is not ideal:

- it is much simpler than ModelNet40
- it is not a standard benchmark used in the original PointNet++ paper
- its objects are digits, not generic 3D object categories

So `3D MNIST` is useful as a secondary validation dataset, but not as a strong realism benchmark.


### 11.3 What the Cross-Dataset Result Actually Shows

The cross-dataset result is modest but meaningful.

What it shows:

- `attn-v2` is not universally worse than PointNet++ SSG
- on a simpler 10-class 3D classification task, it does slightly outperform the baseline
- the gain is especially visible in class accuracy

What it does not show:

- it does not prove that `attn-v2` is a better general-purpose replacement for PointNet++ SSG
- it does not prove robustness on real scanned objects
- it does not prove state-of-the-art performance

Best interpretation:

- `attn-v2` seems capable of learning balanced multi-class decision boundaries on an additional dataset
- but the improvement is too small and the dataset is too easy to claim a major generalization advantage


## 12. Drawbacks and Limitations

### 12.1 Main Method Drawbacks

1. `attn-v2` does not beat the local ModelNet40 baseline on the primary benchmark metric.

2. `attn-v2` is much larger than the baseline.

3. The strongest improvements are in class accuracy, not in instance accuracy.

4. The architecture became significantly more complicated.

5. The biggest parameter cost comes from the global `SA3` hybrid pooling block, which may not be the best place to spend model capacity.


### 12.2 Experimental Drawbacks

1. The ModelNet40 runs were not fully controlled across batch size.

2. There is no multi-seed evaluation.

3. There is no explicit ablation isolating:

- masked local attention only
- hybrid pooling only
- improved classifier only
- stronger cross-level fusion only

4. The second dataset, `3D MNIST`, is easy and not a standard PointNet++ benchmark.

5. Earlier ModelNet40 runs rely on text logs only, not the newer structured `summary.json` format.


### 12.3 Engineering Drawbacks

1. The `ThreeDMNISTDataLoader` reopens the HDF5 file inside `__getitem__`.

2. This is a safe design for HDF5 on macOS, but it is not the fastest design.

3. Versioning full checkpoints and raw dataset files caused GitHub push failures, so large artifacts had to be excluded from git and preserved locally.


## 13. Recommended Next Improvements

If the project is extended, these are the highest-value next steps.

1. Run controlled multi-seed comparisons on ModelNet40 with identical batch size for baseline and `attn-v2`.

2. Ablate the `attn-v2` model into:

- masked local attention only
- hybrid pooling only
- cross-level fusion only
- stronger classifier only

3. Replace `sa3` hybrid pooling with a lighter alternative or revert `SA3` to max pooling.

4. Reduce parameter count by shrinking the `sa3.pool.fuse` block.

5. Evaluate on a more convincing second dataset such as `ScanObjectNN`.


## 14. Reproducibility and Artifact Locations

### 14.1 Key Model Files

- baseline: `models/pointnet2_cls_ssg.py`
- SE: `models/pointnet2_cls_ssg_se.py`
- attention v1: `models/pointnet2_cls_ssg_attn.py`
- attention v2: `models/pointnet2_cls_ssg_attn_v2.py`

### 14.2 Key Training / Testing Files

- training pipeline: `train_classification.py`
- evaluation pipeline: `test_classification.py`
- `3D MNIST` loader: `data_utils/ThreeDMNISTDataLoader.py`

### 14.3 Key Experiment Directories

- ModelNet40 baseline: `log/classification/pointnet2_cls_ssg/`
- ModelNet40 SE: `log/classification/se/`
- ModelNet40 attention v1: `log/classification/attn/`
- ModelNet40 attention v2: `log/classification/attnv2/`
- 3D MNIST baseline: `log/classification/3dmnist_pointnet2_ssg/`
- 3D MNIST attention v2: `log/classification/3dmnist_attnv2/`

### 14.4 Commands Used

ModelNet40 `attn-v2`:

```bash
./.venv/bin/python train_classification.py --model pointnet2_cls_ssg_attn_v2 --epoch 100 --batch_size 16 --log_dir attnv2
```

3D MNIST `attn-v2`:

```bash
./.venv/bin/python train_classification.py --model pointnet2_cls_ssg_attn_v2 --dataset 3dmnist --data_path "3D MNIST" --epoch 100 --batch_size 16 --log_dir 3dmnist_attnv2
```

3D MNIST baseline:

```bash
./.venv/bin/python train_classification.py --model pointnet2_cls_ssg --dataset 3dmnist --data_path "3D MNIST" --epoch 100 --batch_size 16 --log_dir 3dmnist_pointnet2_ssg
```


## 15. PPT-Ready Slide Outline

This section is included specifically so a later session can generate slides quickly.

### Slide 1. Project Title

- PointNet++ classification enhancement on ModelNet40 and 3D MNIST

### Slide 2. Project Motivation

- baseline PointNet++ SSG is strong but may miss richer local and cross-level interactions
- goal: test whether attention and channel recalibration can improve classification

### Slide 3. Baseline Method

- PointNet++ SSG architecture
- set abstraction hierarchy
- baseline classifier

### Slide 4. Engineering Prerequisite

- Apple Silicon compatibility
- MPS-safe sampling and masking
- reproducible logging pipeline

### Slide 5. Optimization Path

- baseline
- SE
- attention v1
- attention v2

### Slide 6. Why Attention v1 Failed

- padded neighbor duplication corrupted attention pooling
- cross-level fusion over-compressed features
- classifier head was too weak

### Slide 7. Attention v2 Design

- masked local attention
- hybrid attention + max pooling
- CLS-token cross-level fusion
- stronger classifier
- MPS-safe implementation

### Slide 8. Experimental Setup

- datasets
- no normals
- 1024 points
- 100 epochs
- optimizer and scheduler

### Slide 9. ModelNet40 Results

- show full comparison table
- emphasize: `attn-v2` improves over `attn-v1` but not over baseline

### Slide 10. Comparison with Paper

- original PointNet++ paper numbers
- local baseline vs paper
- `attn-v2` vs paper
- explain why local baseline is the real benchmark

### Slide 11. 3D MNIST Results

- baseline vs `attn-v2`
- same settings
- small but consistent gain for `attn-v2`

### Slide 12. Cross-Dataset Analysis

- ModelNet40: baseline still best on instance accuracy
- 3D MNIST: `attn-v2` slightly better
- attention seems to help class balance more than pure instance accuracy

### Slide 13. Drawbacks

- heavy parameter cost
- no ModelNet40 top-line win
- 3D MNIST is not a strong realism benchmark
- no multi-seed ablation

### Slide 14. Future Work

- proper ablation
- lighter `attn-v2`
- ScanObjectNN evaluation
- controlled multi-seed reruns

### Slide 15. Final Conclusion

- `attn-v2` is a valid correction over `attn-v1`
- it shows some cross-dataset promise
- but it does not yet justify replacing the strong PointNet++ SSG baseline on ModelNet40


## 16. Final Takeaway

The project achieved three concrete outcomes:

- a stable Apple Silicon training pipeline
- a corrected `attn-v2` architecture that fixes the major problems of `attn-v1`
- a reproducible cross-dataset evaluation setup on `3D MNIST`

Scientifically, the result is mixed rather than purely positive.

The most honest conclusion is:

> The attention-based redesign was worthwhile as an architectural investigation, and `attn-v2` is clearly better than `attn-v1`. However, the strongest local PointNet++ SSG baseline remains the best ModelNet40 model in this project. The improved method shows its clearest benefit on class balance and in the secondary `3D MNIST` experiment, but its parameter cost and incomplete ablation remain significant drawbacks.
