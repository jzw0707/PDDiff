"""Vendored diffusion modules.

Submodules are intentionally not imported here. The upstream eager exports pull in
training-only optional dependencies (for example ema_pytorch) that PDDiff does
not use.
"""

__all__ = []
