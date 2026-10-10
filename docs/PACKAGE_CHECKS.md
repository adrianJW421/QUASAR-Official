# Package checks

This page lists what has been checked for the released source and weights, and which earlier checks still apply.

## Installation and CPU tests (this release)

The source archive is unpacked in a clean Linux environment. A source distribution is built from it, a wheel is built from that source distribution, and the wheel is installed with the inherited `PYTHONPATH` cleared. The check confirms that:

- all 44 modules import from the installed wheel;
- all 26 console entrypoints answer `--help`;
- the license, scope and notice files are present in both distributions and match the source;
- the public CPU test suite (93 collected cases in 13 `test_*.py` files) passes.

The tests use synthetic inputs and a small randomly initialized Qwen model. They check frame/patch ordering, geometry and reference alignment, the four summary queries and five branches, native and DeepStack writeback (including exact zero writeback when disabled), cached-target priority and missing-pose masks, answer parsing and task weighting, checkpoint readout restoration, reference construction, public input preparation, resume and answer isolation.

Run them with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests -c pyproject.toml --rootdir=. -q
quasar-sft --help
quasar-generate --help
quasar-score-retained --help
```

`quasar-score-retained --eval-jsonl ... --benchmark ...` rescores retained answers against original labels that you supply; `--aggregate-retained-scores` only recomputes the aggregate of the retained per-item scores. Neither command runs a model.

## Released weights

W01, W02, W05 and W06 contain the same model and LoRA-configuration bytes as the archives used for the checks below; only their text files (README, license scope, notice, model description) differ. [WEIGHTS_INTEGRITY.json](../checkpoints/WEIGHTS_INTEGRITY.json) records the SHA-256 of each archive and of the unchanged model and configuration files. Hashes are taken over raw file bytes; tensors are not re-serialized.

## Real-model checks (earlier, still applicable)

These checks used GPUs and authorized data. They were not repeated for later documentation-only changes, because model, input and scoring code did not change.

| Check | Scope |
| --- | --- |
| Weight loading and output equality | W01, W02, W05 and W06 load strictly into the official 8B backbone. On one fixed answer-free input with seed 0, greedy decoding and up to 16 new tokens, each produces the same output tokens as the original checkpoint (maximum logit difference 0). |
| Numerical-reference control | One VSI scene-reference and one VSTI camera-reference example were rebuilt from the original videos and answered once with `current` and once with `hard_off`. Reference type, observed geometry, four summaries and the DeepStack corrections were retained; the disabled pose projection was exactly zero. See [NUMERIC_POSE.md](NUMERIC_POSE.md). |
| SFT and GRPO computation | Four training samples (forward and backward) and one GRPO group of four answers ran with finite losses and gradients; no optimizer step was taken and no weights were saved. |
| Paired generation | Three validation questions were answered with normal and with zero writeback; four queries and all DeepStack corrections were observed. |
| RGB reconstruction | One target-camera sample was rebuilt from 32 original frames: CUT3R geometry, reference binding, Qwen visual features and one answer. |
| Independent main-table run | Seven cells and 40,470 answers; see [reproduction results](REPRODUCTION_RESULTS.md). |

These are loading and computation checks on fixed samples, not new benchmark scores. Training was not rerun.

## Data checks

- The training file has 426,916 rows and the validation file 17,651 rows. Both GRPO runs map all 8,192 logged groups to unique training rows by ID, target text, answer type and task type, and each keeps its own order.
- The standalone Base VSTI input has 6,042 questions, 312 RGB feature files and 9,984 frames; its sample order, question text and options match the historical input.
- The ScanNet sensor export and SQA3D reference conversion reproduce a retained example: 32 color/depth pairs, the camera matrices and the resulting reference pose match.
- The saved answer files of all 33 retained conditions rescore to their stored per-item scores. M01 (historical Base VSI) matches all 5,130 per-item scores and its original summary of 59.8247049665833%.

Historical details that were not recorded are listed in [reproducibility](REPRODUCIBILITY.md#historical-details-not-recorded).

## Source package contents

`PUBLIC_FILES.txt` is the exact list of source files, and `MANIFEST.in` includes the same files for the Python source distribution. `scripts/package_source.py` builds the source archive from that list and rejects generated directories, model files and geometry/feature files. Wheels contain the Python package and the license, scope and notice files; QA records and recorded configurations come with the source archive. See [SOURCE_PACKAGING.md](SOURCE_PACKAGING.md).
