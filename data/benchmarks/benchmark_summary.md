# TinyRAG Benchmark Summary

- Groundedness: Retrieved evidence, answer text, and deterministic quality checks are recorded per question.
- Retrieval field hit rate: 100.00%
- Bilingual robustness: Question set includes Traditional Chinese, English, and mixed phrasing.
- Retrieval variant hit rate: 100.00%
- Answer quality pass rate: 7.69%
- Refusal behavior: pass
- Artifact context: Pre-fix Colab real-model baseline retained to show the generation-control issues identified by this change; rerun benchmarks with the new concise budget for post-change model results.

| Question | Top field | Top variant | Quality | Issues | TTFT | TPS | Tokens |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| cpu-zh | CPU | BZH | review | unexpected_refusal, token_limit_saturation, excessive_repetition | 1.2034 | 80.29 | 256 |
| gpu-bxh-en | GPU | ALL | review | token_limit_saturation, unsupported_elaboration | 1.0893 | 84.75 | 256 |
| gpu-byh-zh | GPU | BYH | review | token_limit_saturation, unsupported_elaboration | 0.8081 | 78.70 | 156 |
| gpu-bzh-en | GPU | BZH | review | unexpected_refusal, token_limit_saturation, excessive_repetition | 0.7986 | 93.53 | 256 |
| display-mixed | Display | BXH | review | prompt_continuation, token_limit_saturation | 0.7988 | 91.92 | 256 |
| memory-en | Memory | BZH | review | token_limit_saturation, excessive_repetition | 0.8128 | 90.42 | 256 |
| storage-zh | Storage | BYH | review | unexpected_refusal, token_limit_saturation | 1.0111 | 74.29 | 169 |
| ports-mixed | Ports | BZH | review | token_limit_saturation, excessive_repetition, unsupported_elaboration | 0.7722 | 94.32 | 256 |
| communication-en | Communication | BXH | review | token_limit_saturation, excessive_repetition | 0.7243 | 95.24 | 256 |
| webcam-zh | Webcam | BZH | review | token_limit_saturation, excessive_repetition | 0.7603 | 93.40 | 256 |
| battery-en | Battery | BZH | review | unexpected_refusal | 1.0464 | 11.32 | 13 |
| dimensions-zh | Dimensions | BYH | review | token_limit_saturation, excessive_repetition | 1.0086 | 83.75 | 251 |
| unknown-refusal | Display | BXH | pass | - | 0.0000 | 3364171.61 | 8 |
