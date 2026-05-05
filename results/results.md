# Experimental Results

## Quantitative Results

| Metric | Model A: Linear | Model B: C-Abstractor | Model C: C-Abstractor + LoRA |
|---|---:|---:|---:|
| BLEU | 0.0102 | 0.0224 | **0.0245** |
| ROUGE-L | 0.0020 | 0.0060 | **0.0080** |
| METEOR | 0.1446 | 0.2401 | **0.2696** |
| BERTScore F1 | 0.6856 | 0.7056 | **0.7162** |

## Interpretation

The proposed C-Abstractor + LoRA model achieves the best results across all metrics.  
The absolute BLEU and ROUGE-L values are low, which is expected for Kazakh caption generation due to rich morphology and suffix variation.

BERTScore F1 is the most meaningful metric in this setting because it measures contextual semantic similarity rather than exact n-gram overlap.