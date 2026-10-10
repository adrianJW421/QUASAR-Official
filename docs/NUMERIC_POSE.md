# Numerical-reference comparison

The explicit `current` / `hard_off` comparison retains RGB/video content, observed 3D points, reference-type IDs, four spatial summaries and visual/DeepStack writeback. `hard_off` removes the numerical reference matrix vector, reference-relative point/camera channels and the projected pose embedding including its learned bias. It applies to all three reference types and preserves samples already disabled in the original input. Omit the comparison flags to keep the original evaluator behavior.

This is an inference-time control with the same final W01 weights; no model is trained. Public QUASAR VSI preparation remains blocked where its required references are unavailable ([public VSI](PUBLIC_VSI.md)).

## Run two explicit arms

Prepare inputs using [input construction](INPUT_EXAMPLES.md). Use separate output directories with identical input, feature and model paths. `scripts/run_recorded.py` accepts `--numeric-pose-mode current` or `hard_off` only with answer-free QUASAR generation. It retains all recorded model/decoding settings and checks the actual added-weight file through the existing intervention identity check.

```bash
mkdir -p runs/empty_depth
python scripts/run_recorded.py --role decoderaux_grpo_alltask_final_vsti \
  --parameters runs/input/vsti_current_paths.json --inference-only \
  --numeric-pose-mode current --execute
python scripts/run_recorded.py --role decoderaux_grpo_alltask_final_vsti \
  --parameters runs/input/vsti_hard_off_paths.json --inference-only \
  --numeric-pose-mode hard_off --execute
```

Create the JSONs from `configs/numeric_pose_vsti_paths.example.json`; only `OUTPUT_DIR_44` differs. For VSI use `configs/numeric_pose_vsi_paths.example.json` and role `decoderaux_grpo_alltask_final_vsi`. The low-level equivalent is `quasar-generate` with the numerical-reference control flags (named `gate1` in the code) `--gate1-arm current|hard_off --gate1-reference-family all --gate1-expected-checkpoint-sha256 ACTUAL_WEIGHT_DIGEST`, plus the recorded arguments and `--inference-only`. `--verify-numeric-pose` adds same-input and pose-replacement checks before generating the selected answers. Use it on a fixed small input first; it is not a full-benchmark prerequisite.

The historical comparison used `legacy` for VSI and `provided_full` for VSTI, 32 frames, bf16 Qwen/float32 spatial computation, seed 0, greedy decoding, batch 1 and at most 16 new tokens. New public reconstructed references use the public entry's `legacy` reader mode. They are a new input version. To repeat the historical VSTI setting, supply the repaired historical reference records and add `--query-pose-mode provided_full` to the low-level command; publicly reconstructed records are a different input version.

Every answer records `gate1`, `query_pose_protocol` and `intervention_audit`. The latter records the post-mask pose projection, actual writeback magnitude, DeepStack layer count and four-summary count. A raw generated file must contain no gold answer or score.

## Independent scoring and paired uncertainty

The scorer reads the completed two answer files and separately supplied native scoring rows. It rejects missing, duplicated or reordered IDs, changed reference types, incomplete disabling and mixed coordinate identities. The packaged membership defines all 5,130 VSI and 6,042 historical VSTI questions and their fixed subsets; it contains no VSTI QA.

```bash
quasar-score-numeric-pose --dataset vsti \
  --members data/manifests/numeric_pose_members.json \
  --eval-jsonl runs/prepared/spatial/labels.jsonl \
  --current runs/pose_current/predictions.jsonl \
  --hard-off runs/pose_hard_off/predictions.jsonl \
  --query-pose-mode legacy --output-root runs/pose_comparison
```

Set `--query-pose-mode` to the actual mode shared by the two runs; historical VSTI is `provided_full`. For VSI use `--dataset vsi --query-pose-mode legacy` and its matching scoring rows. This command requires the full denominator; a two-question smoke is not a benchmark score. It produces per-item scores, task/subset summaries and paired scene-cluster intervals using the retained scoring and 5,000-resample implementation, seed 20260812. `primary_score` in `RESULTS.tsv` is a fraction; delta/CI columns ending in `_pp` are percentage points. Subsets average only their present task groups. An interval is unavailable if required task coverage disappears in any resample.

## Archived full comparison

The anonymous [results](../results/numeric_pose_20261008/README.md) retain the completed old-input experiment separately from the seven-cell reproduction and the manuscript main table.

| Dataset / subset | N | Current (%) | Hard off (%) | Change (pp) | Paired 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: | --- |
| VSI, eight tasks | 5,130 | 65.370210 | 64.340455 | −1.029754 | [−1.756070, −0.302189] |
| VSI semantic subset | 1,162 | 62.097434 | 61.611652 | −0.485783 | [−2.470869, 1.502479] |
| VSI remaining subset | 3,968 | 66.461135 | 65.250056 | −1.211078 | [−1.968351, −0.439799] |
| VSTI, five tasks | 6,042 | 66.052674 | 61.580706 | −4.471968 | [−5.277288, −3.699060] |
| VSTI historical valid subset | 5,598 | 66.170944 | 61.601418 | −4.569526 | [−5.402403, −3.775678] |
| VSTI remaining subset | 444 | 90.315315 | 90.540541 | +0.225225 | [0.000000, 0.715574] |

Changes are `hard_off − current`. In the 444-item VSTI subset whose reference was already disabled, 443 answers are identical and one changed. The corrected 5,736-question VSTI release is not used here. Limits of the historical inputs are listed in [release scope](RELEASE_SCOPE.md#known-limitations).
