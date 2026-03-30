## Experiment Routing

For CNT model work, the active experiment directory is `experiments/cnt_paper_repro`.

Do not choose `experiments/cnt_loss_compare` for new model, backbone, training, evaluation, or visualization work unless the user explicitly asks for the legacy loss-comparison pipeline.

When a request is ambiguous, prefer `experiments/cnt_paper_repro`.

Reason:
- `cnt_loss_compare` is a legacy experiment line and should not be used as the default CNT model target anymore.
- Recent CNT task-adapted backbone work belongs under `cnt_paper_repro`.
- `cnt_loss_compare` may still be kept temporarily for legacy reference assets and source datasets, so its presence does not mean it is the correct implementation target.
