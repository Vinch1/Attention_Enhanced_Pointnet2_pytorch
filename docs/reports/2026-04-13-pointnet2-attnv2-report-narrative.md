# PointNet++ Attention Enhancement Narrative Report

## Project Scope

This report is a second, presentation-oriented version of the project record. It keeps the full technical content of the earlier detailed report, but rewrites the material in a more narrative style so it can be read directly by a new session and then turned into a PowerPoint deck with minimal restructuring.

The project began from the upstream PointNet++ single-scale grouping classification baseline in `models/pointnet2_cls_ssg.py`. The original repository already contained a solid PyTorch implementation of PointNet and PointNet++, but it was not yet shaped around Apple Silicon training, structured experiment logging, or cross-dataset verification. The work therefore developed along two intertwined tracks. One track was engineering: making the codebase stable on macOS with Apple Silicon and MPS. The other track was architectural: exploring whether attention-style modifications could improve PointNet++ classification.

The final code and experiment history are spread across four important branches. `master` is the upstream baseline. `feat/apple_silicon` is where platform support and the early classification experiments were added. `pointnet2-attn-v2-mac-safe` contains the corrected attention model and its ModelNet40 experiment. `3dmnist-attnv2-training` extends the codebase to a second dataset, `3D MNIST`, and adds persistent run records such as `run_config.json`, `metrics.csv`, and `summary.json`.


## Baseline Method

The baseline model is the PointNet++ SSG classifier defined in `models/pointnet2_cls_ssg.py`. Architecturally, it follows the standard PointNet++ pattern of hierarchical set abstraction. The network samples and groups points into local neighborhoods, extracts local features by pointwise MLPs, and gradually aggregates those features into a global descriptor. The first abstraction layer samples 512 points with a radius of 0.2 and 32 neighbors. The second abstraction layer samples 128 points with a radius of 0.4 and 64 neighbors. The third abstraction layer performs global aggregation and produces a 1024-dimensional global feature. A simple fully connected classifier maps this global feature through `1024 -> 512 -> 256 -> num_class`.

This baseline is important for two reasons. First, it is the method most directly connected to the corresponding PointNet++ paper. Second, it is already strong on ModelNet40, so any modification has to outperform a competent local baseline rather than only outperform the official paper number from 2017.


## SE Version

The first architectural enhancement was the SE model in `models/pointnet2_cls_ssg_se.py`. SE stands for Squeeze-and-Excitation. In this variant, the standard PointNet++ SSG feature extraction pipeline is retained, but channel attention is inserted after the first two set abstraction stages. The SE block performs global average pooling across the point dimension, learns channel weights through two small fully connected layers, and rescales the original feature tensor channel by channel.

The reasoning behind this version is straightforward. PointNet++ already produces strong local features, but not every channel is equally useful. If the network can learn to emphasize informative channels and suppress weaker ones, class discrimination may improve. The SE version also deepens the classifier. Instead of using the baseline two-hidden-layer head, it adopts a stronger pre-activation style structure with more depth and slightly lower dropout. This reflects the idea that once feature quality improves, the classifier should also have enough capacity to use that richer representation.

In practice, the SE model raised class accuracy but did not beat the strongest baseline run on instance accuracy. This result is still meaningful, because it suggests that channel recalibration helps balance the predictions across classes even if it does not yield the highest overall sample-level accuracy.


## Attention v1

The first attention-driven model was implemented in `models/pointnet2_cls_ssg_attn.py`. This version introduced three new ideas at once. The first was `LocalAttentionPooling`, which replaced max pooling in local neighborhoods with learned attention weights. The second was reuse of SE channel attention after the first two abstraction layers. The third was a `CrossLevelAttention` module that pooled each hierarchy level into a token and fused the three tokens by multi-head self-attention.

The design goal of this version was clear: if max pooling is too crude, local attention might preserve finer local structure, and if using only the final global feature is too restrictive, then cross-level fusion might recover useful multi-scale information. On paper, this was a reasonable direction.

However, `attn-v1` failed in practice, and the reasons are technically concrete. The most important issue came from how PointNet++ groups neighbors. The base utility `query_ball_point` in `models/pointnet2_utils.py` fills missing neighbors by duplicating the first valid point. This is harmless for max pooling, because repeating the same value does not change the maximum. It is not harmless for softmax attention. Under softmax, duplicated padding points receive real probability mass and distort the local aggregation. The second problem was that `CrossLevelAttention` compressed each level into a single mean-pooled token and then averaged the attended tokens again, which stripped away too much structure before attention had any real chance to help. The third problem was the classifier head. It effectively behaved too close to a linear classifier because it lacked enough nonlinear transitions after the attention fusion stage.

The poor ModelNet40 results of `attn-v1` confirmed that these were not theoretical nitpicks. The model underperformed the baseline significantly in both instance accuracy and class accuracy.


## Attention v2

The final architectural version is `models/pointnet2_cls_ssg_attn_v2.py`. This model was not designed as a minor extension of `attn-v1`; it was designed as a repair of the mechanisms that had gone wrong.

The first correction was masked local attention. A new grouping path explicitly returns both grouped indices and a validity mask. Softmax is then computed only over valid neighbors. Padding positions are excluded rather than allowed to collect fake attention mass. This is the most principled fix in the whole project, because it directly aligns the implementation with the mathematical assumption that attention should normalize over real elements only.

The second correction was hybrid local pooling. Instead of fully replacing max pooling, `attn-v2` computes both attention pooling and max pooling, then fuses them. This makes the design more conservative. Max pooling in PointNet and PointNet++ is extremely robust, while attention pooling is more expressive but easier to destabilize. The hybrid module keeps the robustness of max pooling and adds learnable weighting on top of it.

The third correction was stronger cross-level fusion. Instead of using only one mean-pooled token per level, `attn-v2` uses both mean and max summaries for the lower levels, introduces a learnable `CLS` token, and keeps a residual path from the strongest global feature `l3`. This means the model no longer destroys spatial hierarchy so early, and it no longer forces every useful signal through an overly compressed bottleneck.

The fourth correction was the classifier head. The final classifier is wider and explicitly nonlinear, using `512 -> 256 -> 128 -> num_class`. This fixes the shallow effective classifier behavior of `attn-v1`.

The final correction was platform-aware numerical safety. Because this model uses masking and attention more aggressively, it includes additional MPS protections such as finite large-negative masking values instead of `-inf`, `torch.where`-based masking, `torch.nan_to_num` after multi-head attention, and aggressive use of `contiguous()` after tensor rearrangements.


## Why Attention v2 Is Better Than Attention v1

The ModelNet40 results show clearly that `attn-v2` is a real improvement over `attn-v1`. The best ModelNet40 instance accuracy improved from `91.392%` to `92.056%`, and class accuracy improved from `86.231%` to `89.597%`. The class-accuracy jump is especially large. This matters because it indicates the revised architecture is making more balanced predictions across categories rather than merely fitting a few dominant classes better.

The corrected design therefore succeeded in its immediate goal: it recovered most of the damage introduced by the first attention attempt. What it did not do is fully justify replacing the PointNet++ SSG baseline on ModelNet40.


## Training and Evaluation Pipeline Changes

The second major body of work was not architectural but infrastructural. Earlier ModelNet40 runs existed only as conventional text logs, which were sufficient for inspection but not ideal for later reuse. In `3dmnist-attnv2-training`, the training and testing scripts were upgraded so experiments became self-describing.

The updated `train_classification.py` now supports `--dataset` and `--data_path`, allowing the same script to train on both ModelNet and `3D MNIST`. Each run records the exact argument set, device, git branch and commit, dataset root, and model metadata in `run_config.json`. Per-epoch metrics are written to `metrics.csv`. Final best results are stored in `summary.json`. The experiment directory also copies the exact model file and scripts used during the run. This means every run becomes a compact snapshot of code, configuration, and results.

The corresponding `test_classification.py` was also upgraded. It can now infer dataset settings from `run_config.json`, which reduces the chance of evaluating a checkpoint on the wrong dataset or with the wrong class count.


## 3D MNIST Integration

The second dataset chosen for practical cross-dataset evaluation was `3D MNIST`. The main reason was accessibility. Unlike ScanObjectNN or raw ShapeNetCore workflows, `3D MNIST` was already available locally as point-cloud-oriented HDF5 files. This made it realistic to complete a second dataset pipeline within the same repository and the same project timeline.

The new dataloader `data_utils/ThreeDMNISTDataLoader.py` reads `train_point_clouds.h5` and `test_point_clouds.h5`. Each sample is stored as an HDF5 group that contains `points`, `normals`, and an image, with the class label stored in a group attribute. The dataloader samples each point cloud down to `num_point`, applies XYZ normalization, optionally concatenates normals, and returns PyTorch-ready arrays. For macOS HDF5 safety, the dataloader reopens the HDF5 file inside `__getitem__`, and the `3D MNIST` training setup forces dataloader workers to zero.

Locally observed dataset statistics show that the training split contains 5000 samples, the test split contains 1000 samples, and the ten classes are reasonably balanced. Point clouds are much denser than the 1024 points used for training, with average raw point counts above twenty thousand, so 1024-point sampling is a meaningful preprocessing step rather than a trivial reshaping operation.


## Evaluation Metrics

The two classification metrics used throughout the project are instance accuracy and class accuracy. Instance accuracy is the percentage of correctly classified samples across the whole test set. This is the primary benchmark metric in PointNet and PointNet++ literature. Class accuracy is the average of per-class accuracies, which helps reveal whether a model is balanced or whether it favors easy or frequent classes.

For the newer pipeline, additional run information is also stored. These include train instance accuracy, test instance accuracy, test class accuracy, learning rate, and global step for every epoch. These extra signals are useful when discussing convergence, overfitting, and whether a model is peaking early or late.


## Parameter Cost

The parameter count of the baseline `pointnet2_cls_ssg` model is `1,475,688`. The SE model increases this to `1,647,616`, which is only an `11.65%` increase over baseline. Attention v1 grows the count to `1,831,683`, a `24.12%` increase. Attention v2 is much larger at `5,951,459` parameters, which is a `303.30%` increase relative to baseline.

This is one of the most important caveats of the final method. `attn-v2` is not a lightweight refinement. It is a much heavier model. The dominant contributor is the late-stage hybrid pooling path, especially `sa3.pool.fuse.0.weight`, which alone contains more than two million parameters. So even though `attn-v2` is the most technically mature attention variant, it comes with a significant efficiency penalty.


## ModelNet40 Experimental Results

The strongest local baseline run on ModelNet40 reached a best instance accuracy of `92.459%` and a best class accuracy of `89.175%`. The best checkpoint appears to have been saved at epoch 68. The final logged training accuracy at the end of the 100-epoch run was `95.488%`, while the final test metrics were lower than the best checkpoint, indicating the usual fluctuation and mild overfitting near the end.

The SE version reached a best instance accuracy of `91.976%` and a best class accuracy of `89.626%`. This means SE did not beat the baseline on the primary instance-accuracy benchmark, but it did improve class accuracy. The most reasonable interpretation is that SE made the model more balanced across categories, but not more powerful overall on sample-level top-line accuracy.

Attention v1 reached only `91.392%` instance accuracy and `86.231%` class accuracy. This was a clear underperformance. Relative to baseline, it lost more than one percentage point of instance accuracy and almost three points of class accuracy. This result justifies the later critique of its architecture and implementation.

Attention v2 reached `92.056%` instance accuracy and `89.597%` class accuracy on ModelNet40. This result is substantially better than attention v1. The model recovered `0.665` percentage points in instance accuracy and `3.367` percentage points in class accuracy relative to attention v1. It also slightly exceeded the baseline in class accuracy by `0.422` points. However, it still trailed the baseline in instance accuracy by `0.403` points.

The clean takeaway from ModelNet40 is therefore mixed. Attention v2 is a genuine improvement over attention v1, and it is competitive with the baseline. But the baseline remains the best model in the project if the main objective is top-line ModelNet40 instance accuracy.


## Comparison with the PointNet++ Paper

The relevant paper is PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space, published at NeurIPS 2017. The official paper reports `90.7%` classification accuracy for PointNet++ without normals and `91.9%` with normals on ModelNet40.

Compared with these official paper numbers, the local baseline retrain in this repository is stronger. The local baseline achieves `92.459%` without normals, which is `1.759` percentage points above the paper’s XYZ-only number and still `0.559` points above the paper’s normal-augmented number. Attention v2 achieves `92.056%`, which is also above the paper’s XYZ-only figure and slightly above the paper’s normal-augmented figure.

This does not mean the project’s new architecture should be credited for beating the paper. The more careful interpretation is that the current codebase, current data handling, and current training practice already form a stronger baseline than the original 2017 setting. For this reason, the most meaningful comparison is not only against the paper result, but against the strongest local PointNet++ SSG retrain.


## Cross-Dataset Performance on 3D MNIST

The cross-dataset comparison on `3D MNIST` is the fairest head-to-head experiment in the project, because both models were run under the same conditions: 100 epochs, batch size 16, no normals, 1024 points, Adam optimizer, and the same HDF5-safe training pipeline on Apple Silicon.

Under this controlled setting, the baseline `pointnet2_cls_ssg` achieved a best instance accuracy of `98.611%` and a best class accuracy of `98.623%`. Attention v2 achieved a best instance accuracy of `98.710%` and a best class accuracy of `98.939%`. The absolute gains are small: about `0.099` percentage points in instance accuracy and `0.316` points in class accuracy. But they are still consistent gains in favor of `attn-v2`.

The convergence behavior is also informative. The baseline learns faster at the very beginning. Its first-epoch test accuracy is much higher than attention v2. Attention v2 starts slower, which is reasonable for a more complex attention-driven architecture. Over time, however, attention v2 catches up and ultimately produces slightly stronger best results, especially on class accuracy. The baseline reaches its best instance accuracy earlier, while attention v2 continues refining class balance later into training.

The most careful interpretation of the `3D MNIST` experiment is that the corrected attention design does have some generality outside ModelNet40. At the same time, the dataset is relatively easy and not a standard benchmark in the original PointNet++ literature. So this result should be described as a useful cross-dataset validation, not as a definitive claim of real-world generalization.


## Main Strengths of the Final Method

The strongest technical contribution of the final method is that it identifies and fixes the main failure modes of the first attention attempt. The masked local attention design is principled and directly corrects a subtle but serious mismatch between PointNet++ grouping and softmax attention. The hybrid local pooling module is a pragmatic design that preserves PointNet++ robustness while still allowing learnable local weighting. The cross-level fusion module is more defensible than the v1 token averaging scheme because it preserves stronger global information and uses richer multi-scale summaries. The logging pipeline is also a real contribution because it transforms ad hoc training runs into durable experimental artifacts.


## Main Weaknesses and Drawbacks

The most important drawback is that `attn-v2` does not beat the strongest local PointNet++ SSG baseline on ModelNet40 instance accuracy. Since ModelNet40 is the main benchmark tied to the corresponding paper, this limits how strong the project’s final performance claim can be.

The second drawback is efficiency. `attn-v2` is much larger than baseline and substantially larger than both SE and attention v1. This makes the final method harder to justify when the measured accuracy gain is small or absent on the main benchmark.

The third drawback is experimental completeness. The project did not run multi-seed evaluations, and there was no full ablation isolating masked attention alone, hybrid pooling alone, stronger classifier alone, or cross-level fusion alone. That means the report can explain which design ideas appear to matter, but it cannot quantify their isolated causal contributions as cleanly as a full paper-style ablation study would.

The fourth drawback is dataset choice. `3D MNIST` was a practical second dataset, but not an ideal realism benchmark. It is valid as a second classification dataset, but weaker than ScanObjectNN for demonstrating real-world robustness.


## What the Project Accomplished

The project accomplished more than a single model tweak. It established a stable Apple Silicon training environment for this repository, produced multiple model variants with clear experimental comparison, identified why the first attention design failed, constructed a corrected attention version, and extended the codebase so the same training script can now support a second point-cloud classification dataset with structured experimental records.

Scientifically, the project ends in a balanced position rather than a purely positive one. The final architecture is a real improvement over the first attention attempt, but the best local baseline remains stronger on the primary benchmark. The new method appears to help class balance and performs slightly better on the second dataset, but it pays for this with much higher parameter count.

That balance is actually useful. It gives a more credible story for a presentation. Instead of claiming a universal win, the project shows a full research cycle: hypothesis, implementation, failure analysis, redesign, cross-dataset validation, and honest assessment of tradeoffs.


## PPT Slide Outline

### Title Slide

Introduce the project as an investigation into whether attention-based modifications can improve PointNet++ SSG classification, especially on ModelNet40 and an additional dataset.

### Project Motivation

Explain that PointNet++ SSG is already strong, but max pooling and single-level global classification may miss richer local weighting and cross-level information. State that the project tests whether attention and channel recalibration can improve performance.

### Baseline Method

Describe the baseline PointNet++ SSG pipeline. Show the three set abstraction stages and the original classifier head. Emphasize that this is the reference model for all later comparisons.

### Optimization Path

Present the model evolution as baseline, SE, attention v1, and attention v2. Frame attention v2 as a correction of attention v1 rather than just a new variant.

### SE Version

Explain the Squeeze-and-Excitation idea in simple terms: average each channel, learn channel importance, and rescale feature channels. Mention that this improved class accuracy but did not beat baseline instance accuracy.

### Why Attention v1 Failed

Explain the three main issues: invalid padded neighbors corrupting softmax attention, overly compressed cross-level fusion, and a weak classifier head. This slide is important because it shows technical diagnosis, not only empirical failure.

### Attention v2 Design

Describe masked local attention, hybrid attention-plus-max pooling, stronger cross-level fusion with a CLS token and global residual, and the wider classifier head. Mention the MPS-safe implementation details.

### Experimental Setup

Summarize the shared conditions: 1024 points, Adam, 100 epochs, no normals for the main runs, and Apple Silicon MPS. Mention that ModelNet40 comparisons are informative but not perfectly controlled because batch size was not identical across all variants.

### ModelNet40 Results

Show a comparison table of baseline, SE, attention v1, and attention v2. Emphasize that attention v2 strongly improves over attention v1, but baseline still wins on instance accuracy.

### Comparison with the PointNet++ Paper

Show the official PointNet++ paper numbers and compare them with the local retrains. Explain that the local baseline is already stronger than the paper due to codebase and training differences, so the real comparison target is the local baseline.

### Second Dataset: 3D MNIST

Introduce `3D MNIST` as the second classification dataset used for cross-dataset validation. Explain that it was chosen because it was available locally as point-cloud HDF5 files and was easy to integrate into the existing pipeline.

### Cross-Dataset Results

Show the baseline vs attention v2 comparison on `3D MNIST`. Emphasize that attention v2 is slightly better in both instance accuracy and class accuracy under the same settings.

### Interpretation Across Datasets

Summarize the pattern. On ModelNet40, baseline remains best on the main metric. On `3D MNIST`, attention v2 slightly wins. State that the attention redesign appears more helpful for class balance than for top-line benchmark accuracy.

### Drawbacks

State the limitations clearly. Attention v2 is much heavier than baseline, does not beat baseline on ModelNet40 instance accuracy, was not tested with full multi-seed ablation, and was only evaluated on a relatively easy secondary dataset.

### Future Work

Propose the next steps: controlled same-batch-size ModelNet40 reruns, proper ablations, a lighter attention-v2 architecture, and evaluation on a stronger second dataset such as ScanObjectNN.

### Final Conclusion

Conclude that the project produced a technically sound correction of the original attention idea, created a reproducible experimental pipeline, and demonstrated modest cross-dataset promise, but did not replace the PointNet++ SSG baseline on ModelNet40.
