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
from dataclasses import dataclass
from collections import deque
from arc._logging import log as _log

@dataclass
class LLMConfig:
    track_embedding_gradients: bool = True
    embedding_grad_threshold: float = 100.0

    track_attention_patterns: bool = True
    detect_repetition: bool = True
    max_repetition_ratio: float = 0.8

    track_layer_norms: bool = True
    norm_explosion_threshold: float = 1000.0

    track_perplexity: bool = True
    perplexity_spike_threshold: float = 10.0

class LLMHeuristics:
    def __init__(
        self,
        model: nn.Module,
        config: Optional[LLMConfig] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config or LLMConfig()
        self.verbose = verbose

        self._embedding_layer = self._find_embedding_layer()
        self._norm_layers = self._find_norm_layers()

        self._perplexity_history = deque(maxlen=50)
        self._embedding_grad_history = deque(maxlen=50)

        if verbose:
            emb_name = self._embedding_layer[0] if self._embedding_layer else "None"
            _log(f"  LLMHeuristics: Embedding layer={emb_name}, Norms={len(self._norm_layers)}")

    def _find_embedding_layer(self) -> Optional[Tuple[str, nn.Module]]:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                if any(kw in name.lower() for kw in ['embed', 'token', 'wte', 'word']):
                    return (name, module)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                return (name, module)
        return None

    def _find_norm_layers(self) -> List[Tuple[str, nn.Module]]:
        norms = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm if hasattr(nn, 'RMSNorm') else nn.LayerNorm)):
                norms.append((name, module))
            elif 'rmsnorm' in type(module).__name__.lower():
                norms.append((name, module))
        return norms

    def check(
        self,
        logits: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        result = {
            'critical': False,
            'warnings': [],
            'metrics': {},
            'type': None,
        }

        if self.config.track_embedding_gradients and self._embedding_layer:
            _, emb = self._embedding_layer
            if emb.weight.grad is not None:
                emb_grad_norm = emb.weight.grad.norm().item()
                result['metrics']['embedding_grad_norm'] = emb_grad_norm
                self._embedding_grad_history.append(emb_grad_norm)

                if emb_grad_norm > self.config.embedding_grad_threshold:
                    result['critical'] = True
                    result['type'] = 'embedding_gradient_explosion'
                    result['warnings'].append(f"Embedding grad explosion ({emb_grad_norm:.2e})")

        if self.config.detect_repetition:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                if predictions.numel() > 10:
                    last_seq = predictions[0, -20:] if predictions.dim() > 1 else predictions[-20:]
                    unique_ratio = len(torch.unique(last_seq)) / len(last_seq)
                    result['metrics']['unique_token_ratio'] = unique_ratio

                    if unique_ratio < (1 - self.config.max_repetition_ratio):
                        result['warnings'].append(f"Repetitive output detected ({unique_ratio:.2%} unique)")

        if self.config.track_perplexity and loss is not None:
            perplexity = torch.exp(loss).item()
            result['metrics']['perplexity'] = perplexity
            self._perplexity_history.append(perplexity)

            if len(self._perplexity_history) >= 5:
                avg_ppl = sum(list(self._perplexity_history)[-5:-1]) / 4
                if avg_ppl > 0 and perplexity / avg_ppl > self.config.perplexity_spike_threshold:
                    result['critical'] = True
                    result['type'] = 'perplexity_spike'
                    result['warnings'].append(f"Perplexity spike ({perplexity:.2f} vs avg {avg_ppl:.2f})")

        if self.config.track_layer_norms:
            for name, norm in self._norm_layers[:5]:
                if hasattr(norm, 'weight') and norm.weight is not None:
                    norm_val = norm.weight.abs().max().item()
                    if norm_val > self.config.norm_explosion_threshold:
                        result['critical'] = True
                        result['type'] = 'norm_explosion'
                        result['warnings'].append(f"LayerNorm explosion in {name} ({norm_val:.2e})")
                        break

        return result
