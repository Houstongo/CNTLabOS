## Deprecated Experiment Notice

This directory is a legacy CNT loss-comparison experiment and is not the default CNT model target anymore.

Do not use `experiments/cnt_loss_compare` for new CNT model changes, backbone changes, training updates, evaluation updates, or visualization updates unless the user explicitly asks for this legacy pipeline.

For current CNT model work, use `experiments/cnt_paper_repro` instead.

Important:
- The directory may remain temporarily because other experiments still reference assets or datasets under `cnt_loss_compare`.
- That temporary dependency does not mean this model line should receive new work.
- If the user's request is about the active CNT model and does not explicitly name `cnt_loss_compare`, route the task to `cnt_paper_repro`.
