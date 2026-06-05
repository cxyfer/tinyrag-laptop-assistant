# TinyRAG Benchmark Summary

- Groundedness: Retrieved evidence and answer text are recorded per question.
- Exactness hit rate: 100.00%
- Bilingual robustness: Question set includes Traditional Chinese, English, and mixed phrasing.
- Variant awareness hit rate: 100.00%
- Refusal behavior: pass

| Question | Top field | Top variant | TTFT | TPS | Tokens |
| --- | --- | --- | ---: | ---: | ---: |
| cpu-zh | CPU | BZH | 0.0000 | 17589496.50 | 203 |
| gpu-bxh-en | GPU | ALL | 0.0000 | 7591375.80 | 162 |
| gpu-byh-zh | GPU | BYH | 0.0000 | 13599738.12 | 162 |
| gpu-bzh-en | GPU | BZH | 0.0000 | 10859355.56 | 162 |
| display-mixed | Display | BXH | 0.0000 | 16578782.06 | 195 |
| memory-en | Memory | BZH | 0.0000 | 20868992.98 | 207 |
| storage-zh | Storage | BYH | 0.0000 | 15771171.17 | 204 |
| ports-mixed | Ports | BZH | 0.0000 | 24521635.35 | 214 |
| communication-en | Communication | BXH | 0.0000 | 23787817.61 | 209 |
| webcam-zh | Webcam | BZH | 0.0000 | 23204425.61 | 189 |
| battery-en | Battery | BZH | 0.0000 | 29956955.69 | 202 |
| dimensions-zh | Dimensions | BYH | 0.0000 | 24645985.85 | 221 |
| unknown-refusal | Display | BXH | 0.0000 | 5468139.55 | 8 |
