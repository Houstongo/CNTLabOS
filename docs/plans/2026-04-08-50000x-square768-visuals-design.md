# 50000x Square768 Visualization Batch

## Summary
Create a batch of 768x768 square visualization images from the 50000x SEM originals. The output should match the existing paper visualization style: proportional resize and centered placement on a black square canvas. Files are saved as PNG with a `_square768` suffix in a dedicated output folder.

## Goals
- Produce a consistent, paper-ready square visualization for every file in `D:\CNTDATA\coredata\selected_No28_No39_No41_No42\50000`.
- Preserve original files and keep outputs isolated for easy review and selection.

## Non-Goals
- No model inference or segmentation.
- No changes to original images.
- No LaTeX figure updates in this step.

## Inputs
- Source directory: `D:\CNTDATA\coredata\selected_No28_No39_No41_No42\50000`
- File types: PNG images (assumed from current dataset naming).

## Outputs
- Output directory: `D:\CNTDATA\coredata\selected_No28_No39_No41_No42\50000_square768`
- Filename pattern: `<original_name>_square768.png`
- Image size: 768x768, black background, centered image, aspect preserved.

## Processing Rules
- Load each input image.
- Compute scale factor to fit within 768x768 while preserving aspect ratio.
- Resize with high-quality interpolation.
- Paste onto a 768x768 black canvas, centered.
- Save as PNG with suffix `_square768`.

## Verification
- Confirm output count matches input count.
- Spot-check 2-3 outputs for correct centering and aspect ratio.

## Risks
- Extremely large inputs might increase processing time; batch is expected to be manageable.
- If any images are corrupt, log and skip rather than halting the entire run.
