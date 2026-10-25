"""Framework callbacks must stop a poisoned step *before* the optimizer runs
(audit item 10). Each test trains for real with one NaN sample in the data."""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _poisoned_dataset(n=48, poison_index=21):
    torch.manual_seed(0)
    xs, ys = torch.randn(n, 8), torch.randn(n, 1)
    xs[poison_index] = float("nan")
    return xs, ys


def _finite(model):
    return all(torch.isfinite(p).all() for p in model.parameters())


# --- Hugging Face Trainer ----------------------------------------------------

transformers = pytest.importorskip("transformers")


class _HFRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 1)

    def forward(self, x, labels=None):
        out = self.net(x)
        return {"loss": F.mse_loss(out, labels), "logits": out}


class _HFData(torch.utils.data.Dataset):
    def __init__(self):
        self.xs, self.ys = _poisoned_dataset()

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return {"x": self.xs[i], "labels": self.ys[i]}


def _hf_train(tmp_path, callbacks):
    from transformers import Trainer, TrainingArguments

    torch.manual_seed(0)
    model = _HFRegressor()
    args = TrainingArguments(
        output_dir=str(tmp_path), max_steps=12, per_device_train_batch_size=4,
        learning_rate=1e-2, logging_steps=1000, save_strategy="no", report_to=[],
        use_cpu=True, seed=0, disable_tqdm=True,
    )
    Trainer(model=model, args=args, train_dataset=_HFData(), callbacks=callbacks).train()
    return model


def test_hf_nan_batch_poisons_weights_without_arc(tmp_path):
    assert not _finite(_hf_train(tmp_path, []))


def test_hf_callback_skips_poisoned_step_and_rolls_back(tmp_path):
    from arc.api.integrations import HuggingFaceCallback

    callback = HuggingFaceCallback(verbose=False)
    model = _hf_train(tmp_path, [callback])

    assert _finite(model)
    assert callback.controller.total_rollbacks >= 1


# --- PyTorch Lightning ---------------------------------------------------------

def _lightning():
    try:
        import lightning.pytorch as pl
    except ImportError:
        pl = pytest.importorskip("pytorch_lightning")
    return pl


def _lightning_train(callbacks):
    pl = _lightning()

    class Regressor(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(8, 1)

        def training_step(self, batch, _):
            x, y = batch
            return F.mse_loss(self.net(x), y)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-2)

    torch.manual_seed(0)
    module = Regressor()
    xs, ys = _poisoned_dataset()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=4)
    trainer = pl.Trainer(max_steps=12, accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False, callbacks=callbacks)
    trainer.fit(module, loader)
    return module


def test_lightning_nan_batch_poisons_weights_without_arc():
    assert not _finite(_lightning_train([]))


def test_lightning_callback_skips_poisoned_step_and_rolls_back():
    from arc.api.integrations import LightningCallback

    callback = LightningCallback(verbose=False)
    module = _lightning_train([callback])

    assert _finite(module)
    assert callback.controller.total_rollbacks >= 1
