# ARC Deployment Guide

Installation, integration, and the release process for `arc-training`.

## Install

```bash
pip install arc-training                 # from PyPI
pip install "arc-training[hf]"           # + Hugging Face Trainer callback
pip install "arc-training[lightning]"    # + Lightning callback
```

From source, for development:

```bash
git clone https://github.com/a-kaushik2209/ARC.git
cd ARC
pip install -e ".[dev,hf,lightning]"
pytest tests/
```

Requires Python 3.10+, PyTorch 2.4+, NumPy 1.23+, and SciPy 1.9+.

## Integrate

### Plain PyTorch

```python
import arc

controller = arc.Controller(model, optimizer)

for batch in dataloader:
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    if controller.step(loss).should_skip:
        continue
    optimizer.step()
```

### Mixed precision (fp16 with a GradScaler)

```python
controller = arc.Controller(model, optimizer, scaler=scaler)

scaler.scale(loss).backward()
action = controller.step(loss)      # ARC unscales; don't call scaler.unscale_()
if not action.should_skip:
    scaler.step(optimizer)
    scaler.update()
```

### Hugging Face Trainer and Lightning

```python
from arc.api.integrations import HuggingFaceCallback, LightningCallback

Trainer(..., callbacks=[HuggingFaceCallback()])
L.Trainer(..., callbacks=[LightningCallback()])
```

### Configuration

```python
config = arc.ControllerConfig(
    checkpoint_frequency=20,
    baseline_checkpoint_interval=1000,   # your durable-checkpoint cadence, for savings figures
    gpu_hourly_cost=3.0,
    verbose=False,
)
controller = arc.Controller(model, optimizer, scheduler=scheduler, config=config)
```

### Resume after a restart

```python
controller.save_checkpoint("arc_checkpoint.pt")     # model, optimizer, scheduler, scaler, RNG, ARC state
start_step = controller.load_checkpoint("arc_checkpoint.pt")
```

## Docker

```dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN pip install arc-training
```

## Release checklist

1. Update the version in `pyproject.toml` and `arc/__init__.py`.
2. Add release notes to `CHANGELOG.md`.
3. Run `pytest tests/` on a CUDA machine, so the GPU tests run too.
4. Re-run `python experiments/recovery_benchmark.py --seeds 5`, and update the README if the numbers moved.
5. Commit, then tag: `git tag -a v5.2.0 -m "ARC v5.2.0"` and `git push origin master v5.2.0`.
6. Publish to PyPI with one of these:
   - Create a GitHub release for the tag. `.github/workflows/publish.yml` builds the package and uploads it with the `PYPI_API_TOKEN` repository secret.
   - Or build and upload by hand: `python -m build`, `twine check dist/*`, `twine upload dist/*`.
7. Check `pip install arc-training==5.2.0` in a clean environment.

## Package layout

```
arc/            the library (arc.Controller is the entry point)
tests/          pytest suite
examples/       integration examples
experiments/    benchmark and validation scripts
docs/           this guide and historical reports
pyproject.toml  package metadata, dependencies, build configuration
```
