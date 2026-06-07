# TinyRAG Benchmark Summary

- Groundedness: Retrieved evidence, answer text, and deterministic quality checks are recorded per question.
- Retrieval field hit rate: 100.00%
- Bilingual robustness: Question set includes Traditional Chinese, English, and mixed phrasing.
- Retrieval variant hit rate: 100.00%
- Answer quality pass rate: 46.15%
- Refusal behavior: pass

| Question | Top field | Top variant | Quality | Issues | TTFT | TPS | Tokens |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| cpu-zh | CPU | BZH | review | token_limit_saturation | 1.1798 | 49.56 | 96 |
| gpu-bxh-en | GPU | ALL | review | token_limit_saturation | 0.9142 | 58.73 | 96 |
| gpu-byh-zh | GPU | BYH | review | token_limit_saturation | 0.7938 | 63.23 | 96 |
| gpu-bzh-en | GPU | BZH | pass | - | 0.7985 | 45.38 | 60 |
| display-mixed | Display | BXH | pass | - | 0.7716 | 5.66 | 5 |
| memory-en | Memory | BZH | review | token_limit_saturation | 0.7361 | 65.43 | 96 |
| storage-zh | Storage | BYH | pass | - | 0.9206 | 19.92 | 33 |
| ports-mixed | Ports | BZH | review | excessive_repetition | 1.0488 | 53.15 | 95 |
| communication-en | Communication | BXH | pass | - | 0.8052 | 13.30 | 13 |
| webcam-zh | Webcam | BZH | review | token_limit_saturation | 0.8108 | 62.04 | 96 |
| battery-en | Battery | BZH | review | token_limit_saturation, excessive_repetition | 0.7663 | 64.52 | 96 |
| dimensions-zh | Dimensions | BYH | pass | - | 0.7804 | 61.99 | 93 |
| unknown-refusal | Display | BXH | pass | - | 0.0000 | 2511773.97 | 8 |
