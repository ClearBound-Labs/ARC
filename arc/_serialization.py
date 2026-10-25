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

"""The one place ARC deserializes checkpoints.

Loads are always ``weights_only=True``. Unpickling arbitrary objects can run
arbitrary code, so it happens only when the caller opts in explicitly
(``allow_unsafe=True``) for a file they produced themselves. There is no
automatic fallback. See SECURITY.md for the trust boundary.
"""

import os
import warnings
from typing import Any

import torch


def safe_torch_load(path, map_location: Any = None, allow_unsafe: bool = False) -> Any:
    """``torch.load`` that never silently degrades to unsafe unpickling."""
    path = os.fspath(path)
    if not allow_unsafe:
        return torch.load(path, map_location=map_location, weights_only=True)
    warnings.warn(
        f"Loading {path} with weights_only=False (allow_unsafe=True). This can "
        "execute arbitrary code; only do it for checkpoints you produced yourself.",
        UserWarning,
        stacklevel=2,
    )
    return torch.load(path, map_location=map_location, weights_only=False)
