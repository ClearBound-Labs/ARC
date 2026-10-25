# ARC (Automatic Recovery Controller) - Self-Healing Neural Networks
# Copyright (c) 2026 Aryan Kaushik. All rights reserved.
#
# This file is part of ARC.
#
# ARC is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# ARC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ARC. If not, see <https://www.gnu.org/licenses/>.

from typing import Dict, List, Optional, Tuple, Callable
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from collections import deque

@dataclass
class AdversarialAlert:

    is_adversarial: bool
    confidence: float
    detection_method: str
    perturbation_estimate: Optional[float] = None
    recommendation: str = ""

class AdversarialDetector:

    def __init__(
        self,
        model: nn.Module,
        detection_threshold: float = 0.8,
        methods: List[str] = None,
    ):

        self.model = model
        self.detection_threshold = detection_threshold
        self.methods = methods or ["mahalanobis", "lid", "entropy", "gradient"]

        self._layer_means: Dict[str, torch.Tensor] = {}
        self._layer_covs: Dict[str, torch.Tensor] = {}

        self._lid_references: Dict[str, np.ndarray] = {}

        self._entropy_mean: float = 0.0
        self._entropy_std: float = 1.0

        self._grad_mean: float = 0.0
        self._grad_std: float = 1.0

        self._is_fitted = False
        self._hooks = []
        self._activations: Dict[str, torch.Tensor] = {}

    def _register_hooks(self) -> None:

        self._hooks = []
        self._activations = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self._activations[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(make_hook(name))
                self._hooks.append(hook)

    def _remove_hooks(self) -> None:

        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_samples: int = 1000,
    ) -> None:

        self.model.eval()
        device = next(self.model.parameters()).device

        self._register_hooks()

        all_activations: Dict[str, List[torch.Tensor]] = {}
        all_entropies: List[float] = []
        all_grad_mags: List[float] = []

        sample_count = 0

        for batch in dataloader:
            if sample_count >= n_samples:
                break

            x, y = batch[0].to(device), batch[1].to(device)
            x.requires_grad = True

            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

            for name, act in self._activations.items():
                if name not in all_activations:
                    all_activations[name] = []

                flat = act.view(act.size(0), -1)
                all_activations[name].append(flat.cpu())

            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            all_entropies.extend(entropy.detach().cpu().numpy().tolist())

            loss = F.cross_entropy(logits, y)
            # Only x.grad is needed here, but backward() also accumulates into
            # every model parameter with requires_grad=True. Without zeroing,
            # this silently pollutes the caller's optimizer state if they
            # later call optimizer.step() without an explicit zero_grad first.
            self.model.zero_grad(set_to_none=True)
            loss.backward()

            grad_mag = x.grad.view(x.size(0), -1).norm(dim=-1)
            all_grad_mags.extend(grad_mag.cpu().numpy().tolist())
            self.model.zero_grad(set_to_none=True)

            sample_count += x.size(0)

        for name, acts in all_activations.items():
            stacked = torch.cat(acts, dim=0)
            self._layer_means[name] = stacked.mean(dim=0)

            centered = stacked - self._layer_means[name]
            cov = centered.T @ centered / (stacked.size(0) - 1)
            cov = cov + 0.01 * torch.eye(cov.size(0))
            self._layer_covs[name] = cov

            self._lid_references[name] = stacked.numpy()

        self._entropy_mean = np.mean(all_entropies)
        self._entropy_std = max(np.std(all_entropies), 1e-6)

        self._grad_mean = np.mean(all_grad_mags)
        self._grad_std = max(np.std(all_grad_mags), 1e-6)

        self._remove_hooks()
        self._is_fitted = True
        self.model.train()

    def detect(self, x: torch.Tensor) -> AdversarialAlert:

        if not self._is_fitted:
            raise RuntimeError("Must call fit() before detect()")

        single = x.dim() == 3 or (x.dim() == 4 and x.size(0) == 1)
        if single and x.dim() == 3:
            x = x.unsqueeze(0)

        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)
        x.requires_grad = True

        self._register_hooks()

        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)

        scores = {}

        if "mahalanobis" in self.methods:
            maha_score = self._compute_mahalanobis(x)
            scores["mahalanobis"] = maha_score

        if "lid" in self.methods:
            lid_score = self._compute_lid_score(x)
            scores["lid"] = lid_score

        if "entropy" in self.methods:
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropy_z = (entropy.mean().item() - self._entropy_mean) / self._entropy_std

            scores["entropy"] = min(abs(entropy_z) / 3.0, 1.0)

        if "gradient" in self.methods:
            loss = logits.max(dim=-1).values.sum()
            # Same rationale as fit(): backward() accumulates into model
            # parameter .grad as a side effect of computing x.grad; zero it
            # before and after so detect() (routinely called on production
            # inputs) doesn't silently corrupt the caller's optimizer state.
            self.model.zero_grad(set_to_none=True)
            loss.backward()

            grad_mag = x.grad.view(x.size(0), -1).norm(dim=-1).mean().item()
            grad_z = (grad_mag - self._grad_mean) / self._grad_std

            scores["gradient"] = min(max(grad_z, 0) / 3.0, 1.0)
            self.model.zero_grad(set_to_none=True)

        self._remove_hooks()
        self.model.train()

        if scores:
            max_method = max(scores, key=scores.get)
            max_score = scores[max_method]
            avg_score = np.mean(list(scores.values()))
        else:
            max_method = "none"
            max_score = 0.0
            avg_score = 0.0

        is_adversarial = avg_score > self.detection_threshold

        return AdversarialAlert(
            is_adversarial=is_adversarial,
            confidence=avg_score,
            detection_method=max_method,
            perturbation_estimate=max_score if is_adversarial else None,
            recommendation="Reject input" if is_adversarial else "Accept input",
        )

    def _compute_mahalanobis(self, x: torch.Tensor) -> float:

        distances = []

        for name, act in self._activations.items():
            if name not in self._layer_means:
                continue

            flat = act.view(act.size(0), -1).cpu()
            mean = self._layer_means[name]

            max_dim = min(flat.size(-1), 100)
            flat = flat[:, :max_dim]
            mean = mean[:max_dim]
            cov = self._layer_covs[name][:max_dim, :max_dim]

            try:
                cov_inv = torch.linalg.inv(cov)
                centered = flat - mean
                maha = torch.sqrt(torch.sum(centered @ cov_inv * centered, dim=-1))
                distances.append(maha.mean().item())
            except Exception as e:
                warnings.warn(f"Mahalanobis distance computation failed for layer '{name}': {e}")
                continue

        if distances:

            avg_dist = np.mean(distances)
            return min(avg_dist / 100.0, 1.0)

        return 0.0

    def _compute_lid_score(self, x: torch.Tensor) -> float:

        lid_values = []

        for name, act in self._activations.items():
            if name not in self._lid_references:
                continue

            flat = act.view(act.size(0), -1).cpu().numpy()
            ref = self._lid_references[name]

            max_dim = min(flat.shape[-1], 100)
            flat = flat[:, :max_dim]
            ref = ref[:, :max_dim]

            k = min(20, len(ref))

            for sample in flat:
                dists = np.linalg.norm(ref - sample, axis=-1)
                dists = np.sort(dists)[:k]

                if dists[-1] > 0:
                    lid = -k / np.sum(np.log(dists[:-1] / dists[-1] + 1e-10))
                    lid_values.append(lid)

        if lid_values:
            avg_lid = np.mean(lid_values)

            return min(avg_lid / 50.0, 1.0)

        return 0.0

class AdversarialTrainer:

    def __init__(
        self,
        model: nn.Module,
        attack: str = "pgd",
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_steps: int = 7,
    ):

        self.model = model
        self.attack = attack
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def generate_adversarial(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
    ) -> torch.Tensor:

        if self.attack == "fgsm":
            return self._fgsm_attack(x, y, targeted)
        elif self.attack == "pgd":
            return self._pgd_attack(x, y, targeted)
        else:
            raise ValueError(f"Unknown attack: {self.attack}")

    def _fgsm_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool,
    ) -> torch.Tensor:

        x_adv = x.clone().detach().requires_grad_(True)

        logits = self.model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        sign = -1 if targeted else 1
        x_adv = x + sign * self.epsilon * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv.detach()

    def _pgd_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool,
    ) -> torch.Tensor:

        x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

        for _ in range(self.num_steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)

            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            sign = -1 if targeted else 1
            x_adv = x_adv + sign * self.alpha * x_adv.grad.sign()

            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv.detach()

    def adversarial_train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = None,
        mix_ratio: float = 0.5,
    ) -> Tuple[float, float]:

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.train()

        x_adv = self.generate_adversarial(x, y)

        n_adv = int(x.size(0) * mix_ratio)
        x_mixed = torch.cat([x[n_adv:], x_adv[:n_adv]], dim=0)
        y_mixed = torch.cat([y[n_adv:], y[:n_adv]], dim=0)

        optimizer.zero_grad()

        logits = self.model(x_mixed)
        loss = criterion(logits, y_mixed)
        loss.backward()

        optimizer.step()

        with torch.no_grad():
            clean_loss = criterion(self.model(x), y).item()
            adv_loss = criterion(self.model(x_adv), y).item()

        return clean_loss, adv_loss

class RandomizedSmoothing:

    def __init__(
        self,
        model: nn.Module,
        sigma: float = 0.25,
        n_samples: int = 100,
        n_certify: int = 10000,
    ):

        self.model = model
        self.sigma = sigma
        self.n_samples = n_samples
        self.n_certify = n_certify

    def predict(self, x: torch.Tensor) -> Tuple[int, float]:

        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        counts = {}

        with torch.no_grad():
            for _ in range(self.n_samples):
                noise = torch.randn_like(x) * self.sigma
                noisy = x + noise

                logits = self.model(noisy)
                pred = logits.argmax(dim=-1).item()

                counts[pred] = counts.get(pred, 0) + 1

        top_class = max(counts, key=counts.get)
        confidence = counts[top_class] / self.n_samples

        self.model.train()
        return top_class, confidence

    def certify(self, x: torch.Tensor, n_classes: int = 10) -> Tuple[int, float]:

        from scipy.stats import norm

        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        counts = np.zeros(n_classes)

        batch_size = 100
        n_batches = self.n_certify // batch_size

        with torch.no_grad():
            for _ in range(n_batches):
                noise = torch.randn(batch_size, *x.shape[1:], device=device) * self.sigma
                noisy = x + noise

                logits = self.model(noisy)
                preds = logits.argmax(dim=-1).cpu().numpy()

                for pred in preds:
                    counts[pred] += 1

        top_class = int(np.argmax(counts))
        top_count = counts[top_class]

        p_lower = self._lower_confidence_bound(top_count, self.n_certify, 0.001)

        if p_lower > 0.5:
            radius = self.sigma * norm.ppf(p_lower)
        else:
            radius = 0.0

        self.model.train()
        return top_class, radius

    def _lower_confidence_bound(self, k: int, n: int, alpha: float) -> float:

        from scipy.stats import beta
        return beta.ppf(alpha, k, n - k + 1)
