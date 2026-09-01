"""Utilities for running W&B-style sweep YAML configs locally, without a
Weights & Biases account or an active internet connection.

Only ``method: grid`` sweeps are supported, which matches every sweep
config currently checked into ``wandb_configs/cross_validations``. Each
parameter in the sweep YAML must be specified as one of:
  - a fixed value: ``{"value": x}``
  - a list of values to grid-search over: ``{"values": [x, y, z]}``
  - a nested parameter group (used for dict-valued config keys such as
    ``fusion_network_config``/``regression_task_config``):
    ``{"parameters": {...}}``, resolved recursively.
This is the same format W&B sweep configs use.
"""

import itertools
import os

import yaml
from loguru import logger


def load_sweep_config(path: str) -> dict:
    """Load a W&B-style sweep YAML config file from disk."""
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_param_options(spec) -> list:
    """Resolve a single W&B-style parameter spec into a list of possible
    values for that parameter (length 1 if fixed/nested-fixed, >1 if swept).
    """
    if "value" in spec:
        return [spec["value"]]
    if "values" in spec:
        return spec["values"]
    if "parameters" in spec:
        nested = spec["parameters"]
        nested_keys = list(nested.keys())
        nested_options = [_resolve_param_options(nested[k]) for k in nested_keys]
        return [
            dict(zip(nested_keys, combo))
            for combo in itertools.product(*nested_options)
        ]
    raise ValueError(
        f"Unsupported parameter spec (expected 'value', 'values' or "
        f"'parameters'): {spec}"
    )


def expand_grid_sweep(sweep_cfg: dict) -> list[dict]:
    """Expand a ``method: grid`` sweep config into a list of flat run configs.

    Parameters not being swept (specified with ``value``) are held fixed
    across all runs. Parameters being swept (specified with ``values``) are
    combined via a cartesian product across all swept parameters. Nested
    parameter groups (``parameters``) are resolved recursively into dicts.
    """
    method = sweep_cfg.get("method", "grid")
    if method != "grid":
        raise NotImplementedError(
            f"Local sweeps only support method='grid', got method='{method}'. "
            "Random/bayesian sweeps require the W&B backend (--use_wandb)."
        )

    parameters = sweep_cfg["parameters"]
    keys = list(parameters.keys())
    options = [_resolve_param_options(parameters[k]) for k in keys]

    combos = []
    for combo_values in itertools.product(*options):
        combos.append(dict(zip(keys, combo_values)))
    return combos


def run_local_sweep(
    sweep_yaml_path: str, trainer_fn, wandb_run_dir: str, **trainer_kwargs
) -> None:
    """Run every combination of a local grid sweep sequentially in-process.

    ``trainer_fn`` is called once per run as
    ``trainer_fn(config=combo, wandb_run_dir=wandb_run_dir, use_wandb=False,
    sweep_name=sweep_name, **trainer_kwargs)``.
    """
    sweep_cfg = load_sweep_config(sweep_yaml_path)
    sweep_name = sweep_cfg.get(
        "name", os.path.splitext(os.path.basename(sweep_yaml_path))[0]
    )
    combos = expand_grid_sweep(sweep_cfg)

    logger.info(f"Expanded local sweep '{sweep_name}' into {len(combos)} run(s)")
    for i, combo in enumerate(combos):
        logger.info(f"Running local sweep '{sweep_name}' — run {i + 1}/{len(combos)}")
        trainer_fn(
            config=combo,
            wandb_run_dir=wandb_run_dir,
            use_wandb=False,
            sweep_name=sweep_name,
            **trainer_kwargs,
        )
