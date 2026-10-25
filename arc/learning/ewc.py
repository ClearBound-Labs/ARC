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

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field

@dataclass
class TaskMemory:
    task_id: str
    fisher_diag: Dict[str, torch.Tensor] = field(default_factory=dict)
    optimal_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    importance: float = 1.0
    n_samples: int = 0

class ElasticWeightConsolidation:
    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        fisher_estimation: str = "empirical",
        online: bool = False,
        gamma: float = 0.9,
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_estimation = fisher_estimation
        self.online = online
        self.gamma = gamma

        self.task_memories: List[TaskMemory] = []
        self._current_task_id: Optional[str] = None

        self._online_fisher: Dict[str, torch.Tensor] = {}
        self._online_params: Dict[str, torch.Tensor] = {}

    def consolidate_task(
        self,
        task_id: str,
        dataloader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module] = None,
        n_samples: Optional[int] = None,
    ) -> TaskMemory:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        optimal_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        fisher_diag = self._compute_fisher(dataloader, criterion, n_samples)

        memory = TaskMemory(
            task_id=task_id,
            fisher_diag=fisher_diag,
            optimal_params=optimal_params,
            n_samples=n_samples or len(dataloader.dataset),
        )

        if self.online:
            self._update_online_estimates(fisher_diag, optimal_params)
        else:
            self.task_memories.append(memory)

        self._current_task_id = task_id
        return memory

    def _compute_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        n_samples: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        fisher_diag = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        sample_count = 0

        for batch in dataloader:
            if n_samples and sample_count >= n_samples:
                break

            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None

            device = next(self.model.parameters()).device
            x = x.to(device)
            if y is not None:
                y = y.to(device)

            self.model.zero_grad()
            output = self.model(x)

            if self.fisher_estimation == "empirical":
                if y is not None:
                    loss = criterion(output, y)
                else:
                    loss = criterion(output, output.argmax(dim=-1))
            else:
                probs = torch.softmax(output, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze()
                loss = criterion(output, sampled)

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_diag[name] += param.grad.detach() ** 2

            sample_count += x.size(0)

        for name in fisher_diag:
            fisher_diag[name] /= max(sample_count, 1)

        self.model.train()
        return fisher_diag

    def _update_online_estimates(
        self,
        new_fisher: Dict[str, torch.Tensor],
        new_params: Dict[str, torch.Tensor],
    ) -> None:
        for name in new_fisher:
            if name in self._online_fisher:
                self._online_fisher[name] = (
                    self.gamma * self._online_fisher[name] + new_fisher[name]
                )
                self._online_params[name] = new_params[name].clone()
            else:
                self._online_fisher[name] = new_fisher[name].clone()
                self._online_params[name] = new_params[name].clone()

    def compute_penalty(self) -> torch.Tensor:
        if self.online:
            return self._compute_online_penalty()
        else:
            return self._compute_standard_penalty()

    def _compute_standard_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for memory in self.task_memories:
            for name, param in self.model.named_parameters():
                if name in memory.fisher_diag:
                    fisher = memory.fisher_diag[name]
                    optimal = memory.optimal_params[name]

                    penalty += (fisher * (param - optimal) ** 2).sum()

        return 0.5 * self.lambda_ewc * penalty

    def _compute_online_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if name in self._online_fisher:
                fisher = self._online_fisher[name]
                optimal = self._online_params[name]

                penalty += (fisher * (param - optimal) ** 2).sum()

        return 0.5 * self.lambda_ewc * penalty

    def get_importance_scores(self) -> Dict[str, np.ndarray]:
        importance = {}

        if self.online:
            for name, fisher in self._online_fisher.items():
                importance[name] = fisher.cpu().numpy()
        else:
            for name in self.task_memories[0].fisher_diag if self.task_memories else {}:
                total = None
                for memory in self.task_memories:
                    if name in memory.fisher_diag:
                        if total is None:
                            total = memory.fisher_diag[name].clone()
                        else:
                            total += memory.fisher_diag[name]
                if total is not None:
                    importance[name] = total.cpu().numpy()

        return importance

    @property
    def n_tasks(self) -> int:
        if self.online:
            return 1 if self._online_fisher else 0
        return len(self.task_memories)

    def reset(self) -> None:
        self.task_memories.clear()
        self._online_fisher.clear()
        self._online_params.clear()
        self._current_task_id = None

    def state_dict(self) -> Dict:
        return {
            "lambda_ewc": self.lambda_ewc,
            "online": self.online,
            "gamma": self.gamma,
            "task_memories": [
                {
                    "task_id": m.task_id,
                    "fisher_diag": {k: v.cpu() for k, v in m.fisher_diag.items()},
                    "optimal_params": {k: v.cpu() for k, v in m.optimal_params.items()},
                    "importance": m.importance,
                    "n_samples": m.n_samples,
                }
                for m in self.task_memories
            ],
            "online_fisher": {k: v.cpu() for k, v in self._online_fisher.items()},
            "online_params": {k: v.cpu() for k, v in self._online_params.items()},
        }

    def load_state_dict(self, state: Dict, device: str = "cpu") -> None:
        self.lambda_ewc = state["lambda_ewc"]
        self.online = state["online"]
        self.gamma = state["gamma"]

        self.task_memories = [
            TaskMemory(
                task_id=m["task_id"],
                fisher_diag={k: v.to(device) for k, v in m["fisher_diag"].items()},
                optimal_params={k: v.to(device) for k, v in m["optimal_params"].items()},
                importance=m["importance"],
                n_samples=m["n_samples"],
            )
            for m in state["task_memories"]
        ]

        self._online_fisher = {k: v.to(device) for k, v in state["online_fisher"].items()}
        self._online_params = {k: v.to(device) for k, v in state["online_params"].items()}

class ProgressiveNet:
    def __init__(self, base_model_fn, hidden_size: int = 64):
        self.base_model_fn = base_model_fn
        self.hidden_size = hidden_size
        self.columns: List[nn.Module] = []
        self.lateral_connections: List[nn.ModuleList] = []

    def add_column(self) -> nn.Module:
        new_column = self.base_model_fn()

        for column in self.columns:
            for param in column.parameters():
                param.requires_grad = False

        self.columns.append(new_column)

        if len(self.columns) > 1:
            laterals = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(len(self.columns) - 1)
            ])
            self.lateral_connections.append(laterals)

        return new_column

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        if task_id >= len(self.columns):
            raise ValueError(f"Task {task_id} not yet added")

        output = self.columns[task_id](x)

        if task_id > 0 and task_id <= len(self.lateral_connections):
            laterals = self.lateral_connections[task_id - 1]
            for i, (column, lateral) in enumerate(zip(self.columns[:task_id], laterals)):
                with torch.no_grad():
                    prev_output = column(x)
                output = output + lateral(prev_output)

        return output