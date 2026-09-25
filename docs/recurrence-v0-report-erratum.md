# Recurrence V0 G3 report erratum

Run `129d39646933` is terminal. Its immutable `stages/frozen.json` records `stopped_scientific: uninformative_benchmark`, and `stages/report.json` records a passing G6 report. The report was generated from implementation commit `64c80f7` (code SHA256 `687dd74e9fb942a06dc0efe6017b1852891e759890e175e4e4ddce16c221658e`). Keep those files unchanged.

The generated `report.md` says “recurrent LoRA did not pass the registered quality gate.” That sentence is incorrect for this run: no LoRA arm was trained or evaluated. The correct conclusion is:

> No recurrence-efficacy conclusion: F0 failed the preregistered benchmark-informativeness gate, so adapted arms and sealed tests were not run.

F0 answered 4/600 development examples correctly, with family accuracies 0/200 expressions, 4/200 graphs, and 0/200 lists. Graphs also had 128/200 invalid-format outputs. The registered gate required at least two informative F0 families; none qualified. F1 (0/600) and F2 (9/600) are unadapted frozen-loop diagnostics, not LoRA results. G4 discovery, G5 confirmation, and sealed tests are `not_evaluated`.

The report-template correction was regression-tested after G6. It is representation-only: it does not alter data, scores, gate thresholds, or the signed G3 decision. The original `report.md` SHA256 is `e536eaf991c138d95261e9451008357ff32c6fcda0d5a0e6bd638e1e820a829b`; `report.json` SHA256 is `6cfe2fa40ddfeef4c2b6e1f17e7fa59aeed4cdc1adbf864360e447f5d5d8719b`. These remain the historical artifacts, not corrected replacements.

See [PROJECT_LOG.md](../PROJECT_LOG.md) for earlier implementation-failure runs and the pre-confirmation hash-only sealed-file read that was corrected before this final run.
