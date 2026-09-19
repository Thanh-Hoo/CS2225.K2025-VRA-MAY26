"""Render IDEA retrieval results as a video.

One frame per query: the RGB/NIR/TIR query crops beside their top-K gallery
matches, with green borders for correct identities and red for wrong ones.

Usage:
    python visualize_video.py --config_file configs/RGBNT201/IDEA.yml \
        TEST.WEIGHT './IDEA_RGBNT201/IDEAbest.pth'
"""
import os
import os.path as osp
import argparse

import cv2
import numpy as np
import torch

from config import cfg
from data import make_dataloader
from modeling import make_model
from utils.logger import setup_logger
from utils.metrics import R1_mAP, R1_mAP_eval

# Where each dataset keeps its three modalities, and which subdirs hold test data.
LAYOUTS = {
    'RGBNT201': {'dirs': {'RGB': 'RGB', 'NIR': 'NI', 'TIR': 'TI'},
                 'splits': ['test']},
    'MSVR310': {'dirs': {'RGB': 'vis', 'NIR': 'ni', 'TIR': 'th'},
                'splits': ['query', 'bounding_box_test']},
}

LABEL_W = 96      # left gutter holding the modality names
PAD = 6           # gap between cells
HEADER_H = 64     # title bar
RANK_H = 26       # row of "Rank N" captions
BORDER = 5        # correctness border thickness


def build_path_index(root, dataset):
    """Map each image basename to its full path, per modality."""
    layout = LAYOUTS[dataset]
    index = {key: {} for key in layout['dirs']}
    for split in layout['splits']:
        base = osp.join(root, dataset, split)
        if not osp.isdir(base):
            print("Warning: {} does not exist, skipping.".format(base))
            continue
        for dirpath, _, files in os.walk(base):
            leaf = osp.basename(dirpath)
            for key, dirname in layout['dirs'].items():
                if leaf == dirname:
                    for name in files:
                        index[key].setdefault(name, osp.join(dirpath, name))
    # RGBNT201 keeps all three modalities directly under test/, so the walk above
    # finds them; MSVR310 nests them per identity. Either way we now have names.
    for key, mapping in index.items():
        print("Indexed {} {} images".format(len(mapping), key))
    return index


def load_cell(index, modality, name, cell_w, cell_h):
    """Load one crop, resized to the cell size. Grey placeholder if missing."""
    path = index[modality].get(name)
    if path is None or not osp.isfile(path):
        cell = np.full((cell_h, cell_w, 3), 60, dtype=np.uint8)
        cv2.putText(cell, 'missing', (4, cell_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        return cell
    img = cv2.imread(path)
    if img is None:
        return np.full((cell_h, cell_w, 3), 60, dtype=np.uint8)
    return cv2.resize(img, (cell_w, cell_h))


def rank_order(distmat, q_idx, pids, camids, sceneids, num_query, dataset):
    """Gallery indices sorted by distance, with the protocol's junk removed."""
    order = np.argsort(distmat[q_idx])
    q_pid = pids[q_idx]
    kept = []
    for idx in order:
        g_pid = pids[idx + num_query]
        if dataset == 'MSVR310':
            # MSVR310 protocol: drop same-identity samples from the same scene.
            junk = (g_pid == q_pid) and (sceneids[idx + num_query] == sceneids[q_idx])
        else:
            # Market-style protocol: drop same-identity samples from the same camera.
            junk = (g_pid == q_pid) and (camids[idx + num_query] == camids[q_idx])
        if not junk:
            kept.append(idx)
    return kept


TITLE_SCALE = 0.62


def make_title(dataset, feats, mAP, frame_no, total, q_pid, topk, hits):
    return "{}  |  feats={}  |  mAP {:.1%}  |  query {}/{}  id={}  top-{} hits {}/{}".format(
        dataset, '+'.join(feats), mAP, frame_no, total, q_pid, topk, hits, topk)


def frame_size(topk, cell_w, cell_h, longest_title):
    """Fixed frame size for the whole video; the writer locks it on frame one."""
    (title_w, _), _ = cv2.getTextSize(longest_title, cv2.FONT_HERSHEY_SIMPLEX,
                                      TITLE_SCALE, 1)
    width = max(LABEL_W + (topk + 1) * (cell_w + PAD) + PAD, title_w + 24)
    height = HEADER_H + RANK_H + 3 * (cell_h + PAD) + PAD
    return width, height


def draw_frame(index, names, pids, q_idx, ranked, num_query, cell_w, cell_h,
               dataset, feats, mAP, frame_no, total, size):
    modalities = ['RGB', 'NIR', 'TIR']
    topk = len(ranked)
    width, height = size
    frame = np.full((height, width, 3), 24, dtype=np.uint8)

    q_pid = pids[q_idx]
    hits = sum(1 for idx in ranked if pids[idx + num_query] == q_pid)
    title = make_title(dataset, feats, mAP, frame_no, total, q_pid, topk, hits)
    cv2.putText(frame, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, TITLE_SCALE,
                (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, names[q_idx], (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (150, 150, 150), 1, cv2.LINE_AA)

    y0 = HEADER_H + RANK_H
    for row, modality in enumerate(modalities):
        y = y0 + row * (cell_h + PAD)
        cv2.putText(frame, modality, (12, y + cell_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Query column, marked in blue.
        x = LABEL_W + PAD
        cell = load_cell(index, modality, names[q_idx], cell_w, cell_h)
        cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), (255, 176, 0), BORDER)
        frame[y:y + cell_h, x:x + cell_w] = cell
        if row == 0:
            cv2.putText(frame, 'Query', (x, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 176, 0), 1, cv2.LINE_AA)

        # Ranked gallery columns.
        for col, g_idx in enumerate(ranked):
            x = LABEL_W + PAD + (col + 1) * (cell_w + PAD)
            g_pid = pids[g_idx + num_query]
            correct = g_pid == q_pid
            cell = load_cell(index, modality, names[g_idx + num_query], cell_w, cell_h)
            color = (80, 200, 80) if correct else (60, 60, 220)
            cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), color, BORDER)
            frame[y:y + cell_h, x:x + cell_w] = cell
            if row == 0:
                cv2.putText(frame, 'R{}'.format(col + 1), (x, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="IDEA retrieval video")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--topk", default=10, type=int, help="gallery matches per frame")
    parser.add_argument("--num_queries", default=60, type=int,
                        help="queries to render; -1 for all")
    parser.add_argument("--fps", default=4.0, type=float)
    parser.add_argument("--feats", default="LOCAL",
                        help="comma-separated feature keys used for retrieval")
    parser.add_argument("--out", default="", help="output .mp4 path")
    parser.add_argument("--frames_dir", default="", help="also dump frames as PNG here")
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    dataset = cfg.DATASETS.NAMES
    if dataset not in LAYOUTS:
        raise ValueError("No image layout known for dataset {}".format(dataset))

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("IDEA", output_dir, if_train=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda"

    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(device)
    model.eval()

    if dataset == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    logger.info("Extracting features for visualization")
    for img, pid, camid, camids, target_view, imgpath, text in val_loader:
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            text = {'rgb_text': text['rgb_text'].to(device),
                    'ni_text': text['ni_text'].to(device),
                    'ti_text': text['ti_text'].to(device)}
            camids = camids.to(device)
            sceneids = target_view
            target_view = target_view.to(device)
            feat = model(image=img, text=text, cam_label=camids,
                         view_label=target_view, img_path=imgpath)
            if dataset == "MSVR310":
                evaluator.update((feat, pid, camid, sceneids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))

    feats = [f.strip() for f in args.feats.split(',') if f.strip()]
    cmc, mAP, distmat, pids, camids, _, _ = evaluator.compute(query=feats, gallery=feats)
    logger.info("Retrieval for video uses {} -> mAP {:.1%}, Rank-1 {:.1%}".format(
        feats, mAP, cmc[0]))

    pids = np.asarray(pids)
    camids = np.asarray(camids)
    sceneids = np.asarray(getattr(evaluator, 'sceneids', np.zeros(len(pids), dtype=int)))
    names = evaluator.img_paths

    index = build_path_index(cfg.DATASETS.ROOT_DIR, dataset)

    # cfg.INPUT.SIZE_TEST is (height, width).
    cell_h, cell_w = cfg.INPUT.SIZE_TEST[0] // 2, cfg.INPUT.SIZE_TEST[1] // 2

    total = num_query if args.num_queries < 0 else min(args.num_queries, num_query)
    out_path = args.out or osp.join(output_dir, "rank_video_{}.mp4".format(dataset))
    if args.frames_dir and not osp.exists(args.frames_dir):
        os.makedirs(args.frames_dir)

    # Size the frame for the widest title any query can produce.
    longest_title = make_title(dataset, feats, mAP, total, total,
                               int(pids[:num_query].max()), args.topk, args.topk)
    size = frame_size(args.topk, cell_w, cell_h, longest_title)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             args.fps, size)
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer for " + out_path)
    print("Writing {}x{} video to {}".format(size[0], size[1], out_path))

    for q_idx in range(total):
        ranked = rank_order(distmat, q_idx, pids, camids, sceneids, num_query,
                            dataset)[:args.topk]
        frame = draw_frame(index, names, pids, q_idx, ranked, num_query,
                           cell_w, cell_h, dataset, feats, mAP, q_idx + 1, total, size)
        writer.write(frame)
        if args.frames_dir:
            cv2.imwrite(osp.join(args.frames_dir, "query_{:04d}.png".format(q_idx)), frame)
        if (q_idx + 1) % 10 == 0:
            print("Rendered {}/{} queries".format(q_idx + 1, total))

    writer.release()
    logger.info("Saved retrieval video to {}".format(out_path))


if __name__ == "__main__":
    main()
