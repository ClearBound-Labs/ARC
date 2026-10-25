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
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from arc._logging import log as _log

@dataclass
class DiffusionConfig:
    track_attention_entropy: bool = True
    min_attention_entropy: float = 0.1
    max_attention_entropy: float = 5.0

    track_skip_connections: bool = True
    skip_magnitude_threshold: float = 100.0

    critical_timesteps: List[int] = field(default_factory=lambda: [0, 100, 500, 999])
    track_per_timestep: bool = True

class DiffusionHeuristics:
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DiffusionConfig] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config or DiffusionConfig()
        self.verbose = verbose

        self._attention_layers = self._find_attention_layers()
        self._attention_hooks = []
        self._attention_outputs = {}

        self._entropy_history = deque(maxlen=100)
        self._skip_history = deque(maxlen=100)

        if self.config.track_attention_entropy:
            self._register_attention_hooks()

        if verbose:
            _log(f"  DiffusionHeuristics: Found {len(self._attention_layers)} attention layers")

    def _find_attention_layers(self) -> List[Tuple[str, nn.Module]]:
        attention_layers = []
        for name, module in self.model.named_modules():
            if any(attn in name.lower() for attn in ['attn', 'attention', 'crossattn']):
                if isinstance(module, (nn.MultiheadAttention,)) or hasattr(module, 'to_q'):
                    attention_layers.append((name, module))
        return attention_layers

    def _register_attention_hooks(self):
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self._attention_outputs[name] = output.detach()
            return hook

        for name, layer in self._attention_layers[:5]:
            handle = layer.register_forward_hook(make_hook(name))
            self._attention_hooks.append(handle)

    def _compute_attention_entropy(self) -> float:
        if not self._attention_outputs:
            return -1.0

        entropies = []
        for name, attn_out in self._attention_outputs.items():
            attn_probs = torch.softmax(attn_out.flatten(), dim=-1)
            entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10))
            entropies.append(entropy.item())

        return sum(entropies) / len(entropies) if entropies else -1.0

    def check(
        self,
        output: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        result = {
            'critical': False,
            'warnings': [],
            'metrics': {},
            'type': None,
        }

        if self.config.track_attention_entropy:
            entropy = self._compute_attention_entropy()
            result['metrics']['attention_entropy'] = entropy
            self._entropy_history.append(entropy)

            if entropy >= 0:
                if entropy < self.config.min_attention_entropy:
                    result['critical'] = True
                    result['type'] = 'attention_collapse'
                    result['warnings'].append(f"Attention collapsed (entropy={entropy:.4f})")
                elif entropy > self.config.max_attention_entropy:
                    result['warnings'].append(f"Attention too diffuse (entropy={entropy:.4f})")

        output_max = output.abs().max().item()
        output_mean = output.abs().mean().item()
        result['metrics']['output_max'] = output_max
        result['metrics']['output_mean'] = output_mean

        if output_max > 1e6:
            result['critical'] = True
            result['type'] = 'output_explosion'
            result['warnings'].append(f"Output explosion (max={output_max:.2e})")

        if timesteps is not None and self.config.track_per_timestep:
            t = timesteps.mean().item() if timesteps.numel() > 1 else timesteps.item()
            result['metrics']['timestep'] = t

            if t < 100 and output_mean > 10:
                result['warnings'].append(f"High output at low timestep ({t})")

        self._attention_outputs.clear()

        return result

    def cleanup(self):
        for handle in self._attention_hooks:
            handle.remove()
        self._attention_hooks.clear()

    def __enter__(self) -> 'DiffusionHeuristics':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def __del__(self):
        # Best-effort: if a caller drops the instance without calling
        # cleanup()/using it as a context manager, don't leak the forward
        # hooks on the model.
        self.cleanup()
