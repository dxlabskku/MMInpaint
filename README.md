# MMSInpaint
A multi-level semantic mask inpainting dataset designed for image manipulation detection and localization

## Example Images

![Example Images](examples.svg)

## Dataset Generation

The manipulated images are generated using six inpainting models.
The following table summarizes the distribution of source records and generated manipulated images across the different generators.

| Generator         | Architecture      | Text Prompt | Source Records | Generated Images |
| ----------------- | ----------------- | ----------: | -------------: | ---------------: |
| BrushNet          | Diffusion         |         Yes |         14,245 |           85,470 |
| Kolors-Inpainting | Diffusion         |         Yes |         14,245 |           85,470 |
| FLUX.1-Fill-dev   | Flow              |         Yes |         14,245 |           85,470 |
| OmniPaint         | Flow + LoRA       |          No |         14,245 |           85,470 |
| ZITS++            | CNN + Transformer |         N/A |         14,061 |           84,366 |
| PowerPaint        | Diffusion         |          No |         14,240 |           85,440 |
| **Total**         | --                |          -- |     **85,281** |      **511,686** |

**Table:** Distribution of source records and generated manipulated images across the six inpainting models. The text-prompt column indicates whether additional natural-language prompts are used during generation.

## Data Release

The full dataset will be publicly released after the paper is accepted.
