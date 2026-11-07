from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

try:
    from _bootstrap import add_project_src_to_path
except ModuleNotFoundError:
    from scripts._bootstrap import add_project_src_to_path

add_project_src_to_path()

from tristate_mae.utils.config import load_config, save_config


ABLATIONS = {
    # "full": {"alpha": 0.4, "beta": 0.4, "gamma": 0.2, "masking_mode": "tristate"},
    # "without_spatial_inpainting": {"alpha": 0.0, "beta": 0.6, "gamma": 0.4, "masking_mode": "tristate"},
    # "without_cross_modal_translation": {"alpha": 0.5, "beta": 0.0, "gamma": 0.5, "masking_mode": "tristate"},
    # "without_alignment_anchors": {"alpha": 0.5, "beta": 0.5, "gamma": 0.0, "masking_mode": "tristate"},
    # "only_spatial_inpainting": {"alpha": 0.8, "beta": 0.0, "gamma": 0.2, "masking_mode": "tristate"},
    # "only_cross_modal_translation": {"alpha": 0.0, "beta": 0.8, "gamma": 0.2, "masking_mode": "tristate"},
    "random_spatial": {"masking_mode": "standard", "standard_mask_ratio": 0.75},
    "independent_channel": {"masking_mode": "independent", "standard_mask_ratio": 0.75},
    "alpha1beta7gamma2": {"alpha": 0.1, "beta": 0.7, "gamma": 0.2, "masking_mode": "tristate"},
    "alpha2beta6gamma2": {"alpha": 0.2, "beta": 0.6, "gamma": 0.2, "masking_mode": "tristate"},
    "alpha3beta5gamma2": {"alpha": 0.3, "beta": 0.5, "gamma": 0.2, "masking_mode": "tristate"},
    # "alpha5beta3gamma2": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "masking_mode": "tristate"},
    "alpha6beta2gamma2": {"alpha": 0.6, "beta": 0.2, "gamma": 0.2, "masking_mode": "tristate"},
    # "alpha7beta1gamma2": {"alpha": 0.7, "beta": 0.1, "gamma": 0.2, "masking_mode": "tristate"},
    "alpha1beta1gamma8": {"alpha": 0.1, "beta": 0.1, "gamma": 0.8, "masking_mode": "tristate"},
    "alpha2beta2gamma6": {"alpha": 0.2, "beta": 0.2, "gamma": 0.6, "masking_mode": "tristate"},
    "alpha3beta3gamma4": {"alpha": 0.3, "beta": 0.3, "gamma": 0.4, "masking_mode": "tristate"},
    "alpha4beta4gamma2": {"alpha": 0.4, "beta": 0.4, "gamma": 0.2, "masking_mode": "tristate"},
    "alpha5beta5gamma0": {"alpha": 0.5, "beta": 0.5, "gamma": 0.0, "masking_mode": "tristate"},
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    root = Path(base["training"].get("output_dir", "outputs")) / "ablations"
    root.mkdir(parents=True, exist_ok=True)
    for name, params in ABLATIONS.items():
        cfg = copy.deepcopy(base)
        cfg["model"].update(params)
        cfg["training"]["output_dir"] = str(root / name)
        cfg_path = root / f"{name}.yaml"
        save_config(cfg, cfg_path)
        cmd = [sys.executable, "scripts/train_pretrain.py", "--config", str(cfg_path)]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
