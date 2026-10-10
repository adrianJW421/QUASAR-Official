# Full numerical-reference comparison

These small results describe a completed comparison of the final four-summary QUASAR model using the retained old inputs. Four runs produced 22,344 answers; all IDs, original order, independent per-item scores and disabling audits passed. No source answers, media, reconstructed geometry or features from that run are distributed here.

`RESULTS.tsv` preserves the original per-task, per-question-type and subset results. `ANALYSIS_vsi.json` and `ANALYSIS_vsti.json` preserve counts, paired changes, summaries and scene-cluster bootstrap results, with internal run paths omitted. Primary scores and intervals in JSON are fractions; TSV change/interval columns ending in `_pp` use percentage points. Positive changes mean `hard_off - current > 0`.

VSI: 5,130 questions, eight tasks, fixed 1,162/3,968 subsets. VSTI: historical 6,042 questions, five tasks, fixed 5,598/444 subsets. The corrected 5,736-item release is a separate dataset. Confidence intervals use 5,000 paired scene resamples, seed 20260812; subset scores average only their present tasks.

[Run and score](../../docs/NUMERIC_POSE.md) with the explicit comparison entry. [Known limitations](../../docs/RELEASE_SCOPE.md#known-limitations) explain placeholders, camera selection, the already-disabled subset and historical-input scope. These numbers do not replace the manuscript main table or the separately retained seven-cell reproduction.
