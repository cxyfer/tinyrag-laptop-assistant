# TinyRAG Benchmark Summary

- Groundedness: Retrieved evidence, answer text, and deterministic quality checks are recorded per question.
- Retrieval field hit rate: 100.00%
- Bilingual robustness: Question set includes Traditional Chinese, English, and mixed phrasing.
- Retrieval variant hit rate: 100.00%
- Answer quality pass rate: 84.62%
- Refusal behavior: pass

| Question | Top field | Top variant | Quality | Issues | TTFT | TPS | Tokens |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| cpu-zh | CPU | BZH | pass | - | 1.4580 | 34.14 | 78 |
| gpu-bxh-en | GPU | ALL | pass | - | 0.7913 | 47.55 | 67 |
| gpu-byh-zh | GPU | BYH | review | token_limit_saturation | 0.8354 | 53.44 | 96 |
| gpu-bzh-en | GPU | BZH | pass | - | 1.1194 | 34.50 | 60 |
| display-mixed | Display | BXH | pass | - | 0.8252 | 5.16 | 5 |
| memory-en | Memory | BZH | pass | - | 0.7884 | 37.55 | 46 |
| storage-zh | Storage | BYH | pass | - | 0.7808 | 27.94 | 33 |
| ports-mixed | Ports | BZH | pass | - | 0.7764 | 10.56 | 10 |
| communication-en | Communication | BXH | pass | - | 0.7436 | 20.68 | 34 |
| webcam-zh | Webcam | BZH | pass | - | 0.9346 | 34.88 | 49 |
| battery-en | Battery | BZH | review | token_limit_saturation | 0.8387 | 56.02 | 96 |
| dimensions-zh | Dimensions | BYH | pass | - | 0.8667 | 49.65 | 89 |
| unknown-refusal | Display | BXH | pass | - | 0.0000 | 2841918.28 | 8 |
