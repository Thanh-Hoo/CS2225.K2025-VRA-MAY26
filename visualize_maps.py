"""Render IDEA's CDA offset and attention-heatmap figures.

Enables the (normally dormant) visualization hooks in the CDA module and runs a
handful of test images through the model, saving:
    <out_dir>/off_vis/{RGB,NI,TI}/<image>.png    sampling points + offset arrows
    <out_dir>/attn_vis/{v_,t_}{RGB,NIR,TIR}/<image>.png   attention heatmaps

Usage:
    python visualize_maps.py --config_file configs/RGBNT201/IDEA.yml \
        --num_images 8 TEST.WEIGHT './IDEA_RGBNT201/IDEAbest.pth'
"""
import matplotlib
matplotlib.use('Agg')  # headless rendering; must precede any figure creation

import os
import os.path as osp
import argparse

import torch

from config import cfg
from data import make_dataloader
from modeling import make_model
from modeling.fusion_part import CDA_Module
from utils.logger import setup_logger
from visualize_video import build_path_index


def main():
    parser = argparse.ArgumentParser(description="IDEA offset / attention visualization")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--num_images", default=8, type=int,
                        help="how many test images to render figures for")
    parser.add_argument("--out_dir", default="", help="where to write off_vis/ and attn_vis/")
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not cfg.MODEL.DA:
        raise ValueError("MODEL.DA is False, so there is no CDA module to visualize.")

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("IDEA", output_dir, if_train=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda"

    dataset = cfg.DATASETS.NAMES
    out_dir = args.out_dir or osp.join(output_dir, "cda_vis")
    os.makedirs(out_dir, exist_ok=True)

    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(device)
    model.eval()

    # Arm the visualization hook inside the CDA module.
    CDA_Module.VIS.update({
        'enabled': True,
        'out_dir': out_dir,
        'index': build_path_index(cfg.DATASETS.ROOT_DIR, dataset),
        # PIL wants (width, height); cfg.INPUT.SIZE_TEST is (height, width).
        'img_size': (cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]),
        'max_images': args.num_images,
        'done': 0,
    })

    logger.info("Rendering CDA figures for {} images into {}".format(args.num_images, out_dir))
    for img, pid, camid, camids, target_view, imgpath, text in val_loader:
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            text = {'rgb_text': text['rgb_text'].to(device),
                    'ni_text': text['ni_text'].to(device),
                    'ti_text': text['ti_text'].to(device)}
            camids = camids.to(device)
            target_view = target_view.to(device)
            model(image=img, text=text, cam_label=camids, view_label=target_view,
                  img_path=imgpath)
        if CDA_Module.VIS['done'] >= args.num_images:
            break

    CDA_Module.VIS['enabled'] = False
    for sub in ['off_vis', 'attn_vis']:
        root = osp.join(out_dir, sub)
        if osp.isdir(root):
            for name in sorted(os.listdir(root)):
                count = len(os.listdir(osp.join(root, name)))
                print("{}/{}: {} figures".format(sub, name, count))
    logger.info("CDA figures saved under {}".format(out_dir))


if __name__ == "__main__":
    main()
