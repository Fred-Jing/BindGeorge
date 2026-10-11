# Environment scope

Use Python 3.11 or newer and a dedicated environment. The examples do not
install software into a base environment or download pretrained models.

Core tests need PyTorch (with `torch.load(..., weights_only=True)`), NumPy,
SciPy, scikit-learn and joblib. Install the PyTorch CPU/CUDA build appropriate
for your platform using official instructions. `requirements-core.txt` records
dependency ranges, not an exact tested cross-platform lock.

Graph models additionally need e3nn (`requirements-graph.txt`). External AF2,
AF3, ESMC, dMaSIF and Rosetta extraction environments are separate and not
reconstructed by these requirements. The core suite does not test graph model
imports or those upstream tools.

On shared machines, use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
and `CUDA_VISIBLE_DEVICES=''` for the synthetic CPU tests. Set `TMPDIR` to an
approved scratch directory and `PYTHONDONTWRITEBYTECODE=1` to avoid bytecode
caches. The result-display and release-audit commands use only Python's standard
library and can run before scientific dependencies are installed.

Local checks use an existing CPU-capable research environment and one thread,
with CUDA hidden. A fresh installation has not been tested; the observed
versions below provide a starting point for environment setup.
Do not reuse historical Conda exports containing local build URLs as a portable
installer.

Observed local verification versions (2026-09-07): Python 3.12.13, PyTorch
2.12.0, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.8.0 and joblib 1.5.3.
This records the environment used for CPU tests, not a guarantee that these
exact builds are available on every platform or a complete dependency lock.
