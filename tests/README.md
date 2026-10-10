# CPU verification

The public suite has 13 `test_*.py` files (93 collected cases, including parametrized ones). Inputs are synthetic; the Qwen model is randomly initialized with a 32-component hidden state. Real 8B checks on fixed samples are described in [package checks](../docs/PACKAGE_CHECKS.md).

The suite detects:

- Four-query/five-branch/17-target interface mismatches.
- Frame/patch ordering and geometry/reference alignment errors.
- Missing required reference inputs and incorrect effective control flags.
- Native or DeepStack writeback remaining nonzero in an intended writeback-off condition.
- Loss of cached target priority and incorrect missing-pose masks.
- Wrong answer parsing, task weighting, or duplicate sample IDs.
- Missing/unknown decoder checkpoint readouts, saved-but-disabled SIGReg restoration and the shared GRPO loading parameter contract.
- Changed numerical arrays/targets during cache relocation, repeated-row loss, or answer/target fields leaking into inference preparation.
- A source package whose `MANIFEST.in` differs from `PUBLIC_FILES.txt`.

The separate retained-answer checks verify historical/revised/subset membership, options, all 33 answer conditions, and recorded command binding. M01 adds a comparison against all 5,130 original per-item scores and the original nine-value summary, using both the collected historical scorer and the installed scorer. M01 scoring inputs and licensed training/validation QA are included; VSTI evaluation questions and answers are downloaded from the provider (see [data terms](../data/TERMS.md)).

The installation check builds a wheel from the source distribution in a clean environment, clears inherited `PYTHONPATH`, imports all 44 modules and runs `--help` for all 26 console entrypoints. Real-model GPU checks are separate from these synthetic tests and took no optimizer steps.
