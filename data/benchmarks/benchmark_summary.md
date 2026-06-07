# TinyRAG Benchmark Summary

- Groundedness: Retrieved evidence and answer text are recorded per question.
- Exactness hit rate: 100.00%
- Bilingual robustness: Question set includes Traditional Chinese, English, and mixed phrasing.
- Variant awareness hit rate: 100.00%
- Refusal behavior: pass

| Question | Top field | Top variant | TTFT | TPS | Tokens |
| --- | --- | --- | ---: | ---: | ---: |
| cpu-zh | CPU | BZH | 1.2034 | 80.29 | 256 |
| gpu-bxh-en | GPU | ALL | 1.0893 | 84.75 | 256 |
| gpu-byh-zh | GPU | BYH | 0.8081 | 78.70 | 156 |
| gpu-bzh-en | GPU | BZH | 0.7986 | 93.53 | 256 |
| display-mixed | Display | BXH | 0.7988 | 91.92 | 256 |
| memory-en | Memory | BZH | 0.8128 | 90.42 | 256 |
| storage-zh | Storage | BYH | 1.0111 | 74.29 | 169 |
| ports-mixed | Ports | BZH | 0.7722 | 94.32 | 256 |
| communication-en | Communication | BXH | 0.7243 | 95.24 | 256 |
| webcam-zh | Webcam | BZH | 0.7603 | 93.40 | 256 |
| battery-en | Battery | BZH | 1.0464 | 11.32 | 13 |
| dimensions-zh | Dimensions | BYH | 1.0086 | 83.75 | 251 |
| unknown-refusal | Display | BXH | 0.0000 | 3364171.61 | 8 |
