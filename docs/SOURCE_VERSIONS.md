# Implementation and historical evidence

The package contains the original dependency chain. Packaging adapts import paths, search-path setup and asset locations, and the loaders restore every saved checkpoint component and load CUT3R from a supplied code root. Scientific forward calculations, stage losses, historical target values and cache priority remain unchanged. The retained-answer scoring interface uses the existing prediction parsers and official-like numeric scorer; it does not generate answers.

| Evidence | What is established | Remaining limit |
| --- | --- | --- |
| Saved June SFT/GRPO arguments, startup scripts and logs | Actual training recipe, checkpoint roles and stage settings | See [reproducibility](REPRODUCIBILITY.md#historical-details-not-recorded) |
| August 12 diagnostic execution copy | When checked, the 18 packaged dependency modules were byte-identical to the inspected execution copy before import relocation. They have since received the packaging and runtime changes described above and edits to comments, docstrings and user-facing messages, so the current files are no longer byte-identical to that copy | Diagnostic-source correspondence does not bind the June training executions |
| Main retained answers and original gold | Per-item rescoring and full-precision aggregates | Historical source immutability not proved by rescoring |
| September base VSTI full-run execution copies | 6042 retained answers, five-task score, documented base protocol, and four packaged modules; both original full-run shards contain identical code | A separate base runner; the October independent run matches all 6042 answer texts and scores; not evidence of June training source |
| October independent main-table evaluation | Seven cells, 40,470 generated answers; complete membership and frozen-scorer checks; five exact matches | Two VSTI QUASAR score differences remain open; three VSI input routes remain blocked; historical training-source limits remain |
| Fixed 512 analysis | Five saved answer conditions, 127 scene families, source IDs and strict correctness | A fixed historical subset; not a representative new evaluation |
| Current real 8B runtime | QUASAR SFT step 44,544, GRPO parent step 35,840 and final GRPO step 2,048 strictly loaded; fixed-sample calculations, paired generation and one RGB reconstruction passed on fixed inputs | Verifies the current implementation and environment |

The main model is four learned summary embeddings, four spatial updates and five branches. Its bounded visual correction enters native visual features and each aligned DeepStack path. SFT includes spatial/decoder auxiliary supervision, the recorded answer-distribution KL and SIGReg; all-task GRPO trains LoRA only with the spatial forward path retained. GRPO's policy KL is a separate term.

Basic has one spatial path, four updates and geometry-side reference/pose/occupancy/patch objectives. The separately trained Basic NoFlow has a recorded nonzero raw correction but zero training-time writeback. Its historical saved evaluation CLI flag is false; the available evaluator restores the trained checkpoint flag. The historical effective post-load flag was not recorded and remains unknown.

Saved configuration fields missing historically are listed in `configs/commands.json`. No current default is written into a saved historical configuration to fill such a gap. Later no-token, reference-intervention and numeric options retained by the existing code are explicit optional modes; they do not redefine the packaged four-token model. Historical targets, cache precedence and validation masks remain intact.
