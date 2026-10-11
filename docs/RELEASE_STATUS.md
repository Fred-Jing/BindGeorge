# Release notes

BindGeorge includes readout models, evaluation utilities, aggregate benchmark
results and synthetic tests. `release_sources.txt` records the source files
selected for the initial code release on 2026-09-07.

## Available in this release

- Sequence, single-state, pair-state, geometry, atom and surface readout code.
- Training and analysis scripts from the research workflow.
- The 18-entry aggregate benchmark table.
- Synthetic tests for sequence caches, padding, out-of-fold aggregation,
  single/pair readouts, gradients and state-dictionary round trips.

The serial worker uses the current Python executable, with an optional
`BINDER_GRAPH_PYTHON` override for e3nn. This portability adjustment does not
change the model, loss or fold definitions.

## Reproduction status

The aggregate table can be displayed without scientific dependencies. The
synthetic tests require the core environment described in
[ENVIRONMENT.md](ENVIRONMENT.md).

Full training reproduction also needs the original datasets, fold memberships,
feature caches and reference predictions. The preparation workflow additionally
references protocol and scheduler files that are not in the repository; see
[INPUTS.md](INPUTS.md). It cannot run end to end from this checkout alone.

Upstream feature extraction and pretrained weights are separate dependencies.
No trained binder-classifier checkpoint or general-purpose prediction command
is included. The code is a research release rather than a packaged inference
service.

## Verification record

The initial release record reports eight passing CPU tests in an existing
research environment, plus Python parsing, local documentation-link checks and
an aggregate-table comparison against the source. Those are historical checks,
not a fresh-install or cross-platform guarantee. Recorded environment versions
are listed in [ENVIRONMENT.md](ENVIRONMENT.md).

On 2026-10-11, the documentation refresh was checked with Python parsing for all
32 Python files, local documentation-link validation, aggregate-table validation
and the release-tree audit. The eight dependency-based tests were not rerun in
that review because PyTorch, scikit-learn and joblib were unavailable in the
review environment. No training or feature extraction was run.

The release audit checks for selected excluded artifacts, paths and credential
patterns. It supplements manual review and does not establish licensing or
exhaustive secret detection. A project-level license has not yet been selected;
upstream software retains its own terms.
