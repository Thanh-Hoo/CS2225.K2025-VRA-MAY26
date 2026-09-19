"""Report AW-IDEA's parameter overhead vs. plain IDEA (task requirement #17).

Builds the model twice from the same config -- once with
MODEL.ADAPTIVE_WEIGHTING.ENABLED=False (baseline IDEA) and once with it forced to
True (AW-IDEA) -- and prints the parameter count of each, the AdaptiveModalityWeighting
submodule alone, and the percentage increase.

Must be run on the machine that has the full IDEA dependencies installed
(fvcore, einops, yacs, timm, ftfy, ...) and a CUDA device, since
modeling/meta_arch.py::build_transformer hard-codes `.to("cuda")` for the CLIP
backbone. See docs/AW_IDEA_IMPLEMENTATION_PLAN.md, section 0.

Usage:
    python tools/report_param_overhead.py --config_file configs/RGBNT201/IDEA.yml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from modeling import make_model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_aw_params(model):
    if hasattr(model, 'adaptive_modality_weighting'):
        return sum(p.numel() for p in model.adaptive_modality_weighting.parameters())
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AW-IDEA parameter overhead report")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--num_class", default=171, type=int, help="e.g. RGBNT201 train has 171 identities")
    parser.add_argument("--camera_num", default=4, type=int, help="e.g. RGBNT201 has 4 cameras")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.defrost()
    cfg.MODEL.ADAPTIVE_WEIGHTING.ENABLED = False
    cfg.freeze()
    baseline_model = make_model(cfg, num_class=args.num_class, camera_num=args.camera_num, view_num=0)
    baseline_params = count_params(baseline_model)
    del baseline_model

    cfg.defrost()
    cfg.MODEL.ADAPTIVE_WEIGHTING.ENABLED = True
    cfg.freeze()
    aw_model = make_model(cfg, num_class=args.num_class, camera_num=args.camera_num, view_num=0)
    aw_params = count_params(aw_model)
    aw_only_params = count_aw_params(aw_model)

    print("=" * 60)
    print(f"IDEA baseline parameters:      {baseline_params:,}")
    print(f"AdaptiveModalityWeighting-only parameters: {aw_only_params:,}")
    print(f"AW-IDEA total parameters:      {aw_params:,}")
    print(f"Relative increase:             {100.0 * (aw_params - baseline_params) / baseline_params:.4f}%")
    print("=" * 60)
    if (aw_params - baseline_params) != aw_only_params:
        print("WARNING: total delta does not equal the AW submodule's own parameter "
              "count -- something else in the model changed too, investigate before trusting these numbers.")
