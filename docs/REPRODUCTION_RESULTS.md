# Independent main-table reproduction

The independent run completed on 2026-10-07 from a separate installation, using inputs prepared from the original benchmark videos. It generated 40,470 answers across seven cells. The model, input and scoring logic it used is unchanged in this release; later edits changed only documentation, comments and messages. Its run records carry the identifier `quasar-reviewer-v40-20261005-r1`.

Five cells match both the historical answer text and every per-question score. The two VSTI QUASAR cells have score differences whose causes remain unresolved (`SCORE_DIFFERENCE_OPEN`). Three VSI cells lack historical inputs (`BLOCKED_INPUT`).

## Results

Scores are percentages. Differences are percentage points computed from the unrounded references, not from the displayed two-decimal values.

| Dataset | Role | Answers | Manuscript | Independent run | Difference | Comparison |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| VSTI | Base | 6,042 | 46.31 | 46.31 | 0 | Exact match |
| VSTI | Visual-only SFT | 6,042 | 61.86 | 61.86 | 0 | Exact match |
| VSTI | Visual-only SFT+GRPO | 6,042 | 62.26 | 62.26 | 0 | Exact match |
| VSTI | QUASAR SFT | 6,042 | 65.11 | 65.14 | +0.027851123 | Open difference |
| VSTI | QUASAR SFT+GRPO | 6,042 | 66.19 | 66.03 | −0.153553711 | Open difference |
| VSI | Base | 0 / 5,130 | 59.82 | — | — | Missing historical video-loading protocol |
| VSI | Visual-only SFT | 5,130 | 62.79 | 62.79 | 0 | Exact match |
| VSI | Visual-only SFT+GRPO | 5,130 | 63.62 | 63.62 | 0 | Exact match |
| VSI | QUASAR SFT | 0 / 5,130 | 64.86 | — | — | Missing semantic references |
| VSI | QUASAR SFT+GRPO | 0 / 5,130 | 65.36 | — | — | Missing semantic references |

[Full-precision totals](../results/reproduction_20261007/TABLE12_RESULTS.tsv), [task scores and denominators](../results/reproduction_20261007/TASK_RESULTS.tsv) and [structured results](../results/reproduction_20261007/RESULTS.json) keep the measured values separate from the original experiment groups under `experiments/`. Blocked cells have no score.

## Validation scope

All seven answer files have the complete frozen membership and order, without duplicate or failed records. Rescoring their answer text against the pinned official QA reproduces every stored per-question score and summary. Historical answers were separately rescored with the same program and agree with the unrounded manuscript references.

VSTI denominators are 905 camera–object absolute-distance questions, 839 camera-displacement questions, 913 camera-movement-direction questions, 1,645 object–object-relative-position questions and 1,740 camera–object-relative-distance questions. VSI has 565 counting, 834 absolute-distance, 953 object-size, 288 room-size, 710 relative-distance, 968 relative-direction, 194 route-planning and 618 appearance-order questions. Relative direction averages its 217/378/373-question subtypes equally; the total then averages eight tasks equally.

Before full evaluation the run was checked on fixed samples, including resuming after an interruption. The all-cell entry returned exit code 2 because of the three blocked cells.

## Open score differences

In each of the two cells about one answer in ten differs from the historical run (624 and 661 of 6,042), and per-question scores move up and down in similar numbers. The recorded model and generation settings agree (checkpoint roles, backbone revision, 32 frames, 50,176 pixels, seed 0, greedy decoding, 16 new tokens), and the checked inputs of every changed answer match exactly: question fields, options, frames, reference types and matrices, RGB pixels, camera poses and point maps. The historical feature caches and execution environment could not be compared, so the cause is unknown.

## Missing inputs

Base VSI still requires the March 17–18 historical loader and dependency versions, resolved model revision, actual frame sampling/pixels and complete decoding record. QUASAR VSI requires the original definitions of 1,162 semantic viewpoints and a documented conversion into the corresponding CUT3R coordinate system. These cells stay blocked until such definitions are available and validated; see [reproducibility](REPRODUCIBILITY.md#historical-details-not-recorded).

The source contains no third-party QA, video, geometry or feature caches beyond the records listed in [release scope](RELEASE_SCOPE.md). Official downloads are in the [main-table download guide](../data/TABLE12_DOWNLOADS.md); training downloads are in the [download guide](../data/DOWNLOAD_GUIDE.md#training-questions).
