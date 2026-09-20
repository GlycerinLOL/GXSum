# GXSum

[![CI](https://github.com/GlycerinLOL/GXSum/actions/workflows/ci.yml/badge.svg)](https://github.com/GlycerinLOL/GXSum/actions/workflows/ci.yml)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-GXSum-blue)](https://huggingface.co/datasets/GlycerinLOL/GXSum)
[![License](https://img.shields.io/badge/code-Apache--2.0-green)](LICENSE)

Code for the paper:

> **The Continued Value of Classic Summarization Models: Boosting Performance with High-Quality References**
> Ping-Yen Wu\*, Hsiao-Wei Chou\*, Kuan-Yu Chen
> *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 34, pp. 965–977, 2026.
> DOI: [10.1109/TASLPRO.2026.3659424](https://doi.org/10.1109/TASLPRO.2026.3659424)

\* Equal contribution.

## Overview

The paper asks whether the summarization gap between classic encoder-decoder models and frontier LLMs is a matter of architecture or of supervision quality. The evidence points to the latter. Retraining BART, PEGASUS and BRIO on GPT-4-Turbo-generated references makes them competitive with the LLM that produced those references, and on XSum crowd-sourced human evaluation they are preferred over it — at three orders of magnitude fewer parameters.

This repository holds the code: reference generation, the LLM judge, training and evaluation, and the analysis notebooks. The data and trained models are on HuggingFace.

## Data: GXSum

GXSum ships **`id` + `summary` only** — the source articles are BBC content and are not
redistributed. `id` is the BBC article id used by the public
[`EdinburghNLP/xsum`](https://huggingface.co/datasets/EdinburghNLP/xsum), so attaching the
document is one join. `load_gxsum.py` does it:

```python
from load_gxsum import load_gxsum

ds = load_gxsum("small")    # 20k train — DatasetDict with train / validation / test
ds = load_gxsum("medium")   # 50k train
ds = load_gxsum("large")    # 90k train
```

Rows come back as `{"id", "document", "summary"}`. Both datasets are public; no token is needed.

| Config | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `small` | 19,989 | 1,099 | 11,324 |
| `medium` | 49,962 | 2,748 | 11,324 |
| `large` | 90,532 | 5,494 | 11,324 |

Summaries were generated with `gpt-4-1106-preview` at temperature 0, with a soft length constraint of ±5 tokens around the human reference length and a hard upper bound through `max_tokens`. Every summary passed an automated validation query before being accepted. See the [dataset card](https://huggingface.co/datasets/GlycerinLOL/GXSum) for the full description and license.

## Released checkpoints

Fine-tuned on GXSum. The `base_model` column is what each was initialised from, read from its model card.

| Checkpoint | Base model | GXSum config |
| --- | --- | --- |
| [`GlycerinLOL/LLM_Teached_Pegasus_100k`](https://huggingface.co/GlycerinLOL/LLM_Teached_Pegasus_100k) | `google/pegasus-xsum` | large |
| [`GlycerinLOL/LLM_Teached_Pegasus_50k`](https://huggingface.co/GlycerinLOL/LLM_Teached_Pegasus_50k) | — | medium |
| [`GlycerinLOL/LLM_Teached_Pegasus`](https://huggingface.co/GlycerinLOL/LLM_Teached_Pegasus) | — | small |
| [`GlycerinLOL/LLM_Teached_Bart_100k`](https://huggingface.co/GlycerinLOL/LLM_Teached_Bart_100k) | `facebook/bart-large` | large |
| [`GlycerinLOL/LLM_Teached_Bart_50k`](https://huggingface.co/GlycerinLOL/LLM_Teached_Bart_50k) | — | medium |
| [`GlycerinLOL/LLM_Teached_Bart`](https://huggingface.co/GlycerinLOL/LLM_Teached_Bart) | — | small |
| [`vickt/LLM_Teached_BRIO_XSUM`](https://huggingface.co/vickt/LLM_Teached_BRIO_XSUM) | BRIO (PEGASUS backbone) | — |
| [`GlycerinLOL/LLM_Teached_Pegasus_FS`](https://huggingface.co/GlycerinLOL/LLM_Teached_Pegasus_FS) | `google/pegasus-large` (from scratch) | small |

More variants are listed under [GlycerinLOL](https://huggingface.co/GlycerinLOL).

## Quick start

Runs on CPU, needs no API key, and touches only public assets. Run it from the repo root so `load_gxsum` is importable.

```bash
pip install -r requirements.txt

python - <<'PY'
from load_gxsum import load_gxsum
from transformers import pipeline

ds = load_gxsum("small", split="test").select(range(3))
summarizer = pipeline("summarization", model="GlycerinLOL/LLM_Teached_Pegasus_100k")
for row in ds:
    print("GXSum ref :", row["summary"])
    print("model     :", summarizer(row["document"], max_length=62, num_beams=6)[0]["summary_text"])
    print()
PY
```

## Layout

| File | Purpose |
| --- | --- |
| `load_gxsum.py` | Load GXSum with source documents joined back in from `EdinburghNLP/xsum` |
| `sum_by_gpt.py` | Generate XSum reference summaries with GPT-4-Turbo (async, sharded, exponential-backoff retry) |
| `cnndm_gpt_data_creator.py` | Same, for CNNDM |
| `eval_by_gpt.py` | Score summaries with the six-dimension GPT-4 judge |
| `data_process.ipynb` | Export GXSum splits to BRIO `source` / `target` format |
| `generate_eval_split.ipynb` | Assemble the multi-system ranking CSVs used for human annotation |
| `significant_test.ipynb` | Per-sample ROUGE and BERTScore for paired bootstrap testing |
| `fine-tune.py` | Fine-tune and evaluate BART / PEGASUS / BRIO on GXSum |
| `run_summarization.py` | HuggingFace seq2seq training entry point (vendored example) |
| `speed_test.py` | Benchmark three inference strategies (custom loop, HF pipeline, direct `generate`) |
| `cnndm_create_data_chunk.ipynb` | Dataset chunking and token counting |
| `Inference.ipynb` | Inference scratch |
| `tests/` | Smoke tests; run in CI without GPU or network |

The three `*_by_gpt.py` scripts need `OPENAI_API_KEY` in the environment.

## Development

```bash
uvx ruff check .                                # lint
uvx --with numpy --with pytest pytest           # tests (stubs torch/transformers; runs in <1s)
```

CI runs both on every push and pull request.

## Notes

- **`openai` is pinned to 0.28.** The generation and judge scripts use the legacy pre-1.0 global-client API. Do not upgrade without porting those three scripts.
- **`run_summarization.py` requires `transformers>=4.37.0.dev0`**, while `requirements.txt` pins `4.36.0` for the fine-tuning scripts. Install a newer `transformers` if you need that entry point.
- Model checkpoints, generated predictions, and judge outputs are not tracked in this repository.

## Citation

```bibtex
@article{wu2026gxsum,
  author  = {Wu, Ping-Yen and Chou, Hsiao-Wei and Chen, Kuan-Yu},
  title   = {The Continued Value of Classic Summarization Models: Boosting Performance with High-Quality References},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume  = {34},
  pages   = {965--977},
  year    = {2026},
  doi     = {10.1109/TASLPRO.2026.3659424}
}
```

GitHub also offers a "Cite this repository" button from [`CITATION.cff`](CITATION.cff).

## License

Code: [Apache-2.0](LICENSE). Data: see the [GXSum dataset card](https://huggingface.co/datasets/GlycerinLOL/GXSum#licensing-information).

## Contact

Ping-Yen Wu — brian.92308@gmail.com, or open an issue.
