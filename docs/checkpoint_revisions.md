# Foundation-model checkpoint revisions

Every foundation model in [`src/models_tsfm.py`](../src/models_tsfm.py) is pinned to
an explicit Hugging Face commit. Before this pin, all eleven loads resolved to
whatever `main` pointed at on download day, and nothing in the repository recorded
which commit that was.

| Registry key | Hugging Face model id | Revision (commit) | Parameters |
|---|---|---|---|
| `chronos_mini` | `amazon/chronos-t5-mini` | `bd6a4fde8403b8469acd0abd52852b29dbe61c7b` | 20,456,192 (20.5 M) |
| `chronos_large` | `amazon/chronos-t5-large` | `0e46c9c7e2e9f74b53db0617fdfcfe42a413e54a` | 708,963,328 (709.0 M) |
| `chronos_bolt_mini` | `amazon/chronos-bolt-mini` | `251268337516a88e253628c43e1d26ec577b376b` | 21,236,096 (21.2 M) |
| `chronos_bolt_base` | `amazon/chronos-bolt-base` | `5d9f166d69f47aef3401367a7b842e78fe97b121` | 205,292,928 (205.3 M) |
| `timesfm_200m` | `google/timesfm-1.0-200m-pytorch` | `0581e2c56cb06feb51cfd98fc2b4005b74f7187b` | 203,568,960 (203.6 M) |
| `timesfm_500m` | `google/timesfm-2.0-500m-pytorch` | `dc2443792ce5516872b89b37cf1bc058c3bf0c10` | 498,828,960 (498.8 M) |
| `lag_llama` | `time-series-foundation-models/Lag-Llama` | `72dcfc29da106acfe38250a60f4ae29d1e56a3d9` | 2,449,299 (2.45 M) |
| `moirai_small` | `Salesforce/moirai-1.0-R-small` | `f5bd5d01f0d67de856107d43abdd637380aae0a3` | 13,827,528 (13.8 M) |
| `moirai_base` | `Salesforce/moirai-1.0-R-base` | `4fa939a8800d9da346c0280f3d9aeba0d2d35877` | 91,357,728 (91.4 M) |
| `moirai_large` | `Salesforce/moirai-1.0-R-large` | `b6b84e3fbeb9e6927d4271d625441fb51ab6bbd8` | 310,970,624 (311.0 M) |
| `moirai2_small` | `Salesforce/moirai-2.0-R-small` | `30f43ff08c8494f4943ae1521e9d4e94a0fbb389` | 11,387,208 (11.4 M) |

## How the revisions were recovered

The revisions were **not** recorded at run time — they were reconstructed from the
Hugging Face cache at `HF_HOME=/volume1/no_backup/r1062653/hf_cache`, which the
benchmark scripts point at
([`scripts/run_stride2_remaining.sh`](../scripts/run_stride2_remaining.sh)).

Evidence that these are the commits the May 2026 thesis runs actually used:

- Each repo has **exactly one** snapshot directory, so no second revision was ever
  fetched into this cache.
- Each repo's `refs/main` equals that single snapshot hash.
- Every snapshot directory has an mtime of **2026-05-08 to 2026-05-12**, i.e. on or
  before the final stride-2 pass of 2026-05-12. None postdates it.

That is strong evidence rather than proof: the cache lives on `no_backup` and is the
only record. If it is ever cleared, the revisions in this table become the only
surviving provenance.

## How the parameter counts were obtained

Counted directly from the pinned checkpoints, not copied from any source in the
repository: `model.safetensors` headers were parsed and tensor shapes summed, except
for `timesfm_200m`, `timesfm_500m` and `lag_llama`, whose weights ship as
`torch_model.ckpt` / `lag-llama.ckpt` and were summed via `state_dict()` numels.

**This table is the single source of truth for parameter counts.** Three places in
the repository disagreed with it and with each other:

| Model | This table | `plot_thesis.py` `_PARAMS_M` | `plot_thesis.py` `MODELS_UF` |
|---|---|---|---|
| `chronos_bolt_mini` | 21.2 M | 21 M ✓ | 20 M ✗ |
| `lag_llama` | 2.45 M | 2.45 M ✓ | 2 M ✗ |

A docstring in `src/models_tsfm.py` also described the Moirai-1.0 variants as
"small (91M), -base (311M), -large (1.1B)" — the labels were shifted by one
position. That docstring has been corrected. The two `plot_thesis.py` dicts were
left untouched: they feed published figures, and changing them is a figure change,
not a documentation change.

## What the pin does and does not guarantee

Four of the five loader families accept a `revision` argument directly:
`ChronosPipeline` / `BaseChronosPipeline` (forwarded to `transformers`),
`MoiraiModule` / `Moirai2Module` (explicit parameter), and `hf_hub_download`
(Lag-Llama).

**TimesFM needed a workaround.** `timesfm.TimesFmCheckpoint` has no `revision`
field — its fields are `version, path, huggingface_repo_id, type, step, local_dir`.
However, `TimesFmTorch.load_from_checkpoint` uses `checkpoint.path` verbatim when it
is set, and only falls back to `snapshot_download(repo_id)` when it is `None`. So
`TimesFMForecaster._pinned_ckpt_path()` resolves the pinned commit to a concrete file
with `hf_hub_download(..., revision=...)` and passes it as `path`. The pin is
therefore enforced, not merely documented.

All eleven loads were verified two ways:

1. All eleven load successfully with their pinned revisions.
2. Substituting a deliberately wrong revision makes every family **fail** rather than
   silently fall back to `main` — so the pin is not cosmetic.

One wrinkle worth knowing: a bad revision surfaces as a clear error for Chronos
(`OSError`) and TimesFM (`LocalEntryNotFoundError`), but for Moirai it appears as
`TypeError: MoiraiModule.__init__() missing 7 required positional arguments`. That is
`config.json` failing to resolve, leaving the constructor with no keyword arguments.
It is a revision problem, despite the message.

## Adding or changing a model

Set `_REVISION` on the forecaster class (or add an entry to
`MoiraiForecaster._REVISION_BY_VARIANT`, which is keyed by variant because the repo id
is built from a template), then add a row here. `MoiraiForecaster` raises a
`ValueError` naming this file if a variant has no pinned revision.
