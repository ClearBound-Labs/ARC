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

import torch
import torch.nn as nn
from typing import Dict, Any

from arc.intervention.heuristics.diffusion import DiffusionHeuristics
from arc.intervention.heuristics.llm import LLMHeuristics
from arc.intervention.heuristics.detection import DetectionHeuristics
from arc._logging import log as _log

class AutoHeuristics:

    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose

        self.model_type = self._detect_model_type()

        if self.model_type == 'diffusion':
            self.heuristics = DiffusionHeuristics(model, verbose=verbose)
        elif self.model_type == 'llm':
            self.heuristics = LLMHeuristics(model, verbose=verbose)
        elif self.model_type == 'detection':
            self.heuristics = DetectionHeuristics(model, verbose=verbose)
        else:
            self.heuristics = None
            if verbose:
                _log(f"  AutoHeuristics: Unknown model type, using generic monitoring")

    def _detect_model_type(self) -> str:
        model_name = type(self.model).__name__.lower()
        module_names = [name.lower() for name, _ in self.model.named_modules()]
        module_str = ' '.join(module_names)

        if any(kw in model_name for kw in ['unet', 'diffusion', 'ddpm', 'ldm']):
            return 'diffusion'
        if 'timestep' in module_str or ('down_block' in module_str and 'up_block' in module_str):
            return 'diffusion'

        if any(kw in model_name for kw in ['gpt', 'llama', 'bert', 'transformer', 'lm']):
            return 'llm'
        if 'embed_tokens' in module_str or 'lm_head' in module_str:
            return 'llm'

        if any(kw in model_name for kw in ['yolo', 'rcnn', 'retinanet', 'fcos', 'ssd']):
            return 'detection'
        if 'backbone' in module_str and ('neck' in module_str or 'head' in module_str):
            return 'detection'

        return 'unknown'

    def check(self, output: torch.Tensor, **kwargs) -> Dict[str, Any]:
        if self.heuristics is None:
            return {
                'critical': False,
                'warnings': [],
                'metrics': {'model_type': 'unknown'},
                'type': None,
            }

        return self.heuristics.check(output, **kwargs)

if __name__ == "__main__":
    print("Testing Model-Specific Heuristics...")

    class FakeLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(1000, 256)
            self.layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])
            self.norm = nn.LayerNorm(256)
            self.lm_head = nn.Linear(256, 1000)

        def forward(self, x):
            x = self.embed_tokens(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.lm_head(x)

    model = FakeLLM()
    auto = AutoHeuristics(model)
    print(f"  Detected model type: {auto.model_type}")
    assert auto.model_type == 'llm'

    x = torch.randint(0, 1000, (2, 32))
    out = model(x)
    loss = out.mean()
    loss.backward()

    result = auto.check(out, loss=loss)
    print(f"  Check result: critical={result['critical']}, warnings={len(result['warnings'])}")
    assert 'metrics' in result
    print("OK")
