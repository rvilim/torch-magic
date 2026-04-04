"""Configuration system for MagIC PyTorch port.

Call ``configure(overrides)`` BEFORE importing any other magic_torch module.
If not called, params.py falls back to environment variables for backward
compatibility with existing test runners.

Usage::

    from magic_torch.config import configure
    configure({"l_max": 32, "ra": 1e6})
    from magic_torch.main import run  # now uses l_max=32, ra=1e6
"""

# Module-level config dict. Populated by configure() or left empty
# (in which case params.py falls back to env vars).
_config: dict = {}


def configure(overrides: dict | None = None):
    """Set simulation configuration.

    Must be called before any other magic_torch module is imported,
    because params.py reads configuration at import time.

    Args:
        overrides: dict of config keys to values. Keys use snake_case
            matching Fortran namelist names (l_max, n_r_max, ra, etc.).
            Only keys that differ from defaults need to be specified.
    """
    global _config
    _config = {}
    if overrides:
        _config.update(overrides)
