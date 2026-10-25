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

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
from collections import deque

@dataclass
class PersistencePair:
    dimension: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        return self.death - self.birth

    @property
    def midpoint(self) -> float:
        return (self.birth + self.death) / 2

@dataclass
class TopologicalState:
    betti_0: int
    betti_1: int
    betti_2: int
    total_persistence: float
    max_persistence: float
    topological_complexity: float
    landscape_fragmentation: float

class LossLandscapeSampler:
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 100,
        radius: float = 0.1,
        n_directions: int = 20,
    ):
        self.model = model
        self.n_samples = n_samples
        self.radius = radius
        self.n_directions = n_directions

        self._direction_cache: List[Dict[str, torch.Tensor]] = []
        self._samples_cache: List[Tuple[float, np.ndarray]] = []

    def sample(
        self,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        directions = self._generate_directions()

        original_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

        coordinates = []
        losses = []

        try:
            if len(directions) >= 2:
                # Genuine 2D loss-landscape scan: pair up directions and move
                # along a (x, y) grid spanned by each pair, so the resulting
                # coordinates actually vary in both dimensions instead of
                # every sample collapsing onto y=0 regardless of direction.
                grid = np.linspace(-self.radius, self.radius, 5)
                pairs = list(zip(directions[0::2], directions[1::2]))

                for d1, d2 in pairs:
                    for x in grid:
                        for y in grid:
                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    if name in d1:
                                        param.data = (
                                            original_params[name]
                                            + x * d1[name]
                                            + y * d2[name]
                                        )

                            with torch.no_grad():
                                output = self.model(inputs)
                                loss = loss_fn(output, targets).item()

                            coordinates.append([x, y])
                            losses.append(loss)
            else:
                radii = np.linspace(0, self.radius, 5)
                for direction in directions[:self.n_directions]:
                    for r in radii:
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if name in direction:
                                    param.data = original_params[name] + r * direction[name]

                        with torch.no_grad():
                            output = self.model(inputs)
                            loss = loss_fn(output, targets).item()

                        coordinates.append([r])
                        losses.append(loss)
        finally:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in original_params:
                        param.data = original_params[name]

        return np.array(coordinates), np.array(losses)

    def _generate_directions(self) -> List[Dict[str, torch.Tensor]]:
        directions = []

        for _ in range(self.n_directions):
            direction = {}
            total_norm_sq = 0.0

            for name, param in self.model.named_parameters():
                d = torch.randn_like(param)
                direction[name] = d
                total_norm_sq += (d ** 2).sum().item()

            norm = math.sqrt(total_norm_sq)
            for name in direction:
                direction[name] /= norm

            directions.append(direction)

        return directions

class PersistentHomology:
    def __init__(self, max_dimension: int = 2):
        self.max_dimension = max_dimension
        if max_dimension > 0:
            warnings.warn(
                "PersistentHomology only computes dimension-0 persistence "
                "(connected components via union-find); higher-dimensional "
                f"features (dimension 1..{max_dimension}) are not implemented, "
                "so betti_1/betti_2 in TopologicalState will always be 0.",
                stacklevel=2,
            )

    def compute(
        self,
        points: np.ndarray,
        values: np.ndarray,
    ) -> List[PersistencePair]:
        n_points = len(values)

        order = np.argsort(values)
        sorted_values = values[order]

        parent = list(range(n_points))
        rank = [0] * n_points

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        pairs = []
        birth_times = {i: sorted_values[i] for i in range(n_points)}
        if len(points.shape) == 1:
            points = points.reshape(-1, 1)

        edges = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                birth = max(sorted_values[i], sorted_values[j])
                edges.append((birth, i, j))

        edges.sort()

        component_birth = {i: sorted_values[i] for i in range(n_points)}

        for birth, i, j in edges:
            pi, pj = find(i), find(j)
            if pi != pj:
                if component_birth[pi] < component_birth[pj]:
                    pairs.append(PersistencePair(
                        dimension=0,
                        birth=component_birth[pj],
                        death=birth,
                    ))
                    del component_birth[pj]
                else:
                    pairs.append(PersistencePair(
                        dimension=0,
                        birth=component_birth[pi],
                        death=birth,
                    ))
                    del component_birth[pi]

                union(i, j)

        for comp, birth in component_birth.items():
            pairs.append(PersistencePair(
                dimension=0,
                birth=birth,
                death=float('inf'),
            ))

        finite_pairs = [p for p in pairs if p.death != float('inf')]

        return finite_pairs

    def compute_betti_numbers(
        self,
        pairs: List[PersistencePair],
        threshold: float,
    ) -> Tuple[int, int, int]:
        betti = [0, 0, 0]

        for pair in pairs:
            if pair.birth <= threshold < pair.death:
                if pair.dimension < 3:
                    betti[pair.dimension] += 1

        return tuple(betti)

class TopologicalFeatures:
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        radius: float = 0.05,
    ):
        self.model = model
        self.sampler = LossLandscapeSampler(
            model, n_samples=n_samples, radius=radius
        )
        self.homology = PersistentHomology(max_dimension=2)

        self._states: deque = deque(maxlen=50)

    def compute(
        self,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> TopologicalState:
        coords, losses = self.sampler.sample(loss_fn, inputs, targets)

        pairs = self.homology.compute(coords, losses)

        state = self._extract_features(pairs, losses)
        self._states.append(state)

        return state

    def _extract_features(
        self,
        pairs: List[PersistencePair],
        losses: np.ndarray,
    ) -> TopologicalState:
        if len(losses) > 0:
            threshold = np.median(losses)
            betti = self.homology.compute_betti_numbers(pairs, threshold)
        else:
            betti = (0, 0, 0)

        persistences = [p.persistence for p in pairs if p.persistence < float('inf')]

        if persistences:
            total_persistence = sum(persistences)
            max_persistence = max(persistences)

            probs = np.array(persistences) / (total_persistence + 1e-10)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            total_persistence = 0.0
            max_persistence = 0.0
            entropy = 0.0

        fragmentation = betti[0] / (len(losses) + 1)

        return TopologicalState(
            betti_0=betti[0],
            betti_1=betti[1],
            betti_2=betti[2],
            total_persistence=total_persistence,
            max_persistence=max_persistence,
            topological_complexity=entropy,
            landscape_fragmentation=fragmentation,
        )

    def get_trend(self) -> Dict[str, float]:
        if len(self._states) < 3:
            return {}

        recent = list(self._states)[-10:]

        betti_0s = [s.betti_0 for s in recent]
        complexities = [s.topological_complexity for s in recent]

        return {
            "betti_0_trend": np.polyfit(range(len(betti_0s)), betti_0s, 1)[0],
            "complexity_trend": np.polyfit(range(len(complexities)), complexities, 1)[0],
        }