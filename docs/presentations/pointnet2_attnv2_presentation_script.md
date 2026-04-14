# PointNet++ Attention Enhancement Presentation Script

## Slide 1 - Title
Today I will present our PointNet++ attention enhancement project. The core question behind this work was whether attention-based modifications could improve PointNet++ single-scale grouping classification, on two datasets.

## Slide 2 - Project Scope
The project one goal. That is architectural exploration. I wanted to see whether channel attention, local attention pooling, and cross-level fusion could improve PointNet++ classification behavior.

## Slide 3 - Baseline Method
The baseline model is the PointNet++ SSG classifier. It uses three set abstraction stages to move from local neighborhoods to a global feature, and then classifies from a 1024-dimensional descriptor. This baseline is already strong, even have stronger number than the original paper.

## Slide 4 - Optimization Path
The architectural path was baseline, then SE, then attention-v1, then attention-v2. The SE model added channel recalibration and a stronger classifier. Attention-v1 introduced local attention pooling and cross-level fusion. Attention-v2 was not just another variant. It was specifically designed to correct the places where attention-v1 was technically unsound.

## Slide 5 - Why SE Was Added
The SE version applies Squeeze-and-Excitation after the first two abstraction layers. The principle is that not every feature channel is equally useful, so the network should learn which channels to emphasize and which to suppress. In practice, this improved class accuracy, which suggests better class balance, but it did not surpass the baseline on top-line instance accuracy.

## Slide 6 - Why Attention v1 Failed
Attention-v1 failed for three main reasons. First, PointNet++ radius grouping duplicates the first valid neighbor to fill padded slots, which is safe for max pooling but corrupts softmax attention. Second, the cross-level fusion compressed too much information before attention had any chance to help. Third, the classifier head was effectively too weak. These problems explain why the first attention model fell below baseline.

## Slide 7 - Attention v2 Design
Attention-v2 fixed those issues directly. It introduced masked local attention so softmax is applied only over valid neighbors. It fused attention pooling with max pooling so the model keeps the robustness of PointNet++ while adding learnable local weighting. It also used a stronger cross-level fusion design with mean and max summaries, a learnable CLS token, and a residual path from the global feature. Finally, it used a stronger nonlinear classifier head.

## Slide 8 - Engineering and Logging
The final pipeline also improved the surrounding experiment system. Device selection is now unified across MPS, CUDA, and CPU. The training script supports multiple datasets through a shared interface. Every run stores configuration, per-epoch metrics, and a final summary JSON. This matters because later analysis and presentation depend on reproducible artifacts, not only console logs.

## Slide 9 - Experimental Setup
For this experiment, we used Adam optimizer, an initial learning rate of 0.001, StepLR scheduling, and no normal vectors. On 3D MNIST, the baseline and attention-v2 comparison used exactly the same setup.

## Slide 10 - ModelNet40 Results
On ModelNet40, the strongest local baseline reached 92.459 percent instance accuracy and 89.175 percent class accuracy. The SE model improved class accuracy slightly but not instance accuracy. Attention-v1 underperformed clearly. Attention-v2 recovered most of that lost ground and improved strongly over attention-v1, especially on class accuracy, but it still did not surpass the baseline on instance accuracy. So the final answer on ModelNet40 is mixed: attention-v2 is much better than v1, but baseline remains best on the primary metric.

## Slide 11 - Comparison with the PointNet++ Paper
The original PointNet++ paper reported 90.7 percent without normals and 91.9 percent with normals on ModelNet40. Our local baseline is stronger than both of those numbers even without normals, which shows that the modern codebase and training setup already form a stronger baseline than the original paper configuration. That means the correct comparison target for our new method is the local retrain, not just the paper result.

## Slide 12 - 3D MNIST Integration
To verify the method on another dataset, we integrated 3D MNIST. This dataset was chosen because it was available locally in point-cloud-oriented HDF5 files and could be added to the repository quickly. A dedicated dataloader was written to sample each object down to 1024 points, normalize coordinates, and optionally include normals. The pipeline also forces zero dataloader workers on macOS for HDF5 safety.

## Slide 13 - 3D MNIST Results
On 3D MNIST, the baseline achieved 98.611 percent instance accuracy and 98.623 percent class accuracy. Attention-v2 slightly outperformed it with 98.710 percent instance accuracy and 98.939 percent class accuracy. The gain is small, but it is consistent and appears more clearly in class accuracy. The convergence curves also show that the baseline learns faster early, while attention-v2 improves more gradually and ends slightly higher.

## Slide 14 - Drawbacks and Future Work
The main limitation is that attention-v2 does not beat the strongest baseline on ModelNet40 instance accuracy. The second limitation is efficiency. Attention-v2 is much larger than the baseline. There is also no full ablation study and no multi-seed evaluation. Finally, while 3D MNIST is a valid second classification dataset, it is not as strong a realism benchmark as ScanObjectNN. The most valuable next steps would be controlled same-batch-size reruns, lighter attention-v2 blocks, full ablation, and evaluation on a stronger second dataset.

## Slide 15 - Final Conclusion
The project produced a stable Apple Silicon training pipeline, a corrected attention architecture, and a reusable cross-dataset evaluation workflow. The scientific result is honest and balanced. Attention-v2 is clearly better than attention-v1 and shows some promise on a second dataset, but it does not replace the strong PointNet++ SSG baseline on ModelNet40. That balance gives us a credible presentation story built around design, diagnosis, evidence, and limitations.
