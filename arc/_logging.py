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

"""ARC's console output goes through the ``arc`` logger.

Components log only when the user asked for verbose output. If the
application has configured logging, ARC's records flow into that
configuration; otherwise a plain stderr handler is attached so ``verbose=True``
still shows something. Silence ARC with
``logging.getLogger("arc").setLevel(logging.WARNING)``.
"""

import logging

logger = logging.getLogger("arc")


def enable_console_logging() -> None:
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("ARC: %(message)s"))
        logger.addHandler(handler)


def log(message, level: int = logging.INFO) -> None:
    """Emit one verbose message (callers guard on their own verbose flag)."""
    enable_console_logging()
    logger.log(level, message)
