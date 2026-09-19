"""Print (and optionally plot) AW-IDEA's per-sample modality weights.

Reuses the repo's own dataloader/model factories (data.make_dataloader,
modeling.make_model) -- no dataset- or model-specific code duplicated here.

Must be run on a machine with the full IDEA dependencies + a CUDA device (see
docs/AW_IDEA_IMPLEMENTATION_PLAN.md, section 0), with MODEL.ADAPTIVE_WEIGHTING.ENABLED
true in the given config/checkpoint.

Usage:
    python tools/visualize_modality_weights.py \
        --config_file configs/RGBNT201/AW_lightweight.yml \
        --weight ./IDEA_RGBNT201_AW/IDEAbest.pth \
        --num_samples 20 \
        --plot reports/modality_weights_RGBNT201.png
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from data import make_dataloader
from modeling import make_model


def main():
    parser = argparse.ArgumentParser(description="Visualize AW-IDEA modality weights")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--weight", required=True, type=str, help="path to an AW-IDEA checkpoint")
    parser.add_argument("--num_samples", default=20, type=int, help="number of query samples to print")
    parser.add_argument("--plot", default="", type=str, help="optional path to save an aggregate bar chart (PNG)")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not cfg.MODEL.ADAPTIVE_WEIGHTING.ENABLED:
        print("WARNING: MODEL.ADAPTIVE_WEIGHTING.ENABLED is False in this config -- "
              "there will be no modality weights to report.")

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.eval()
    model.load_param(args.weight)
    model.to("cuda")

    all_rgb, all_nir, all_tir, all_names = [], [], [], []
    printed = 0
    with torch.no_grad():
        for img, pid, camid, camids, target_view, imgpath, text in val_loader:
            img = {'RGB': img['RGB'].to("cuda"), 'NI': img['NI'].to("cuda"), 'TI': img['TI'].to("cuda")}
            text = {'rgb_text': text['rgb_text'].to("cuda"), 'ni_text': text['ni_text'].to("cuda"),
                    'ti_text': text['ti_text'].to("cuda")}
            camids = camids.to("cuda")
            target_view = target_view.to("cuda")
            model(image=img, text=text, cam_label=camids, view_label=target_view, img_path=imgpath)
            weights = model.last_modality_weights
            if weights is None:
                print("model.last_modality_weights is None -- adaptive weighting is disabled for this checkpoint/config.")
                return
            rgb, nir, tir = weights['rgb'].cpu(), weights['nir'].cpu(), weights['tir'].cpu()
            for i, name in enumerate(imgpath):
                if printed < args.num_samples:
                    print(f"Sample {name}\n  RGB : {rgb[i].item():.2f}\n  NIR : {nir[i].item():.2f}\n  TIR : {tir[i].item():.2f}")
                    printed += 1
                all_rgb.append(rgb[i].item())
                all_nir.append(nir[i].item())
                all_tir.append(tir[i].item())
                all_names.append(name)
            if printed >= args.num_samples and args.plot == "":
                break

    def mean_std(values):
        t = torch.tensor(values)
        return t.mean().item(), t.std().item()

    rgb_mean, rgb_std = mean_std(all_rgb)
    nir_mean, nir_std = mean_std(all_nir)
    tir_mean, tir_std = mean_std(all_tir)
    print("\nAggregate over {} samples:".format(len(all_rgb)))
    print(f"  RGB : {rgb_mean:.2f} +/- {rgb_std:.2f}")
    print(f"  NIR : {nir_mean:.2f} +/- {nir_std:.2f}")
    print(f"  TIR : {tir_mean:.2f} +/- {tir_std:.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(['RGB', 'NIR', 'TIR'], [rgb_mean, nir_mean, tir_mean],
               yerr=[rgb_std, nir_std, tir_std], capsize=6,
               color=['#d62728', '#2ca02c', '#1f77b4'])
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='uniform (alpha=1)')
        ax.set_ylabel('alpha (modality weight)')
        ax.set_title(f'AW-IDEA modality weights ({cfg.DATASETS.NAMES}, n={len(all_rgb)})')
        ax.legend()
        os.makedirs(os.path.dirname(args.plot) or '.', exist_ok=True)
        fig.savefig(args.plot, bbox_inches='tight', dpi=150)
        print(f"Saved aggregate bar chart to {args.plot}")


if __name__ == '__main__':
    main()
