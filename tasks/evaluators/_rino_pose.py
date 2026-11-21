"""RINO human-pose estimation protocol: crop, letterbox, skeleton rendering, colour-free decoding, metrics.

Ported from RINO (arXiv:2607.12450), github.com/yangtiming/RINO @ c37075b,
tasks/human_pose_estimation/evaluate/{data.py, codec.py, metrics.py}. MIT License,
Copyright (c) 2026 RINO Authors. The crop, letterbox, decoder and metrics are unchanged;
`render_skeleton` generalises RINO's visualisation `encode_pose` (joint radius / bone width are
parameters) so fine-tuning targets can be drawn on the model canvas.

Protocol (top-down, one sample per person): the person's GT box is expanded by 25% per side and
cropped, the crop is letterboxed into a fixed 512x768 black canvas, the editor repaints it as an
OpenPose-style skeleton on black, and COCO-17 keypoints are decoded colour-free (distance-transform
peaks + template matching). Scores: raw PCK (crop pixels), PA PCK (after a robust scale+translation
fit) normalised by the longer GT box side, micro-averaged over visible joints, plus OKS.

Shared by tasks/dataloaders/coco_pose_estimation.py, tasks/evaluators/pose_skeleton.py,
tasks/aggregators/pose_skeleton.py and finetune_data (training targets), so all use one definition.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

CANVAS = (512, 768)          # (W, H) fixed model canvas
MARGIN = 0.25                # expand bbox by 25% per side before cropping
MIN_KPTS = 8                 # need enough visible joints to score meaningfully
MIN_AREA = 64 * 64           # skip tiny people (undecodable)

PROMPT = (
    "Convert this photo of a person into an OpenPose human pose skeleton on a solid black background. "
    "Draw each body joint as a small solid colored dot and connect adjacent joints with colored bones, "
    "following the standard OpenPose color convention (head and shoulders in warm reds and oranges, "
    "arms in greens, legs in blues and purples). Place every dot exactly on the corresponding body part "
    "of the person. Remove the person, the clothing, the background and all texture, and output only "
    "the bright colored skeleton on pure black."
)

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
N_COCO = 17
BODY = list(range(5, 17))            # 12 body joints (head joints 0..4 handled apart)
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]
_OPENPOSE = np.array([
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85],
], dtype=np.float32)
_COCO_TO_OP = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
COCO_COLORS = _OPENPOSE[_COCO_TO_OP]
TEMPLATE = np.array([
    [0.50, 0.06], [0.46, 0.05], [0.54, 0.05], [0.42, 0.06], [0.58, 0.06],
    [0.62, 0.22], [0.38, 0.22], [0.68, 0.38], [0.32, 0.38], [0.70, 0.53],
    [0.30, 0.53], [0.58, 0.56], [0.42, 0.56], [0.58, 0.76], [0.42, 0.76],
    [0.58, 0.96], [0.42, 0.96],
])
OKS_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                       1.07, 1.07, .87, .87, .89, .89]) / 10.0
PCK_THRESHOLDS = (0.05, 0.1, 0.2)


# --- selection and crop (RINO data.py) -------------------------------------------------------
def keep_annotation(ann: dict) -> bool:
    return (not ann.get("iscrowd", 0) and ann.get("num_keypoints", 0) >= MIN_KPTS
            and ann.get("area", 0) >= MIN_AREA)


def crop_person(img: Image.Image, bbox_xywh, keypoints):
    """Crop to the bbox expanded by MARGIN. Returns (crop, kpts_in_crop (17,3), (bbox_w, bbox_h), box)."""
    W, H = img.size
    x, y, w, h = (float(v) for v in bbox_xywh)
    mx, my = w * MARGIN, h * MARGIN
    x0, y0 = max(0, int(round(x - mx))), max(0, int(round(y - my)))
    x1, y1 = min(W, int(round(x + w + mx))), min(H, int(round(y + h + my)))
    crop = img.crop((x0, y0, x1, y1))
    kpts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3).copy()
    vis = kpts[:, 2] > 0
    kpts[vis, 0] -= x0
    kpts[vis, 1] -= y0
    return crop, kpts, (w, h), (x0, y0, x1, y1)


# --- letterbox (RINO codec.py) ----------------------------------------------------------------
def letterbox(crop: Image.Image, size: tuple[int, int] = CANVAS):
    """Fit `crop` into a fixed (W,H) black canvas, keeping aspect ratio. Returns (canvas, scale, ox, oy)."""
    W, H = size
    cw, ch = crop.size
    scale = min(W / cw, H / ch)
    nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas.paste(crop.resize((nw, nh), Image.BILINEAR), (ox, oy))
    return canvas, float(scale), int(ox), int(oy)


def letterbox_kpts(kpts: np.ndarray, scale: float, ox: int, oy: int) -> np.ndarray:
    """Crop coords -> canvas coords (inverse of unletterbox_kpts)."""
    out = np.asarray(kpts, dtype=np.float32).copy()
    f = out[:, 2] > 0
    out[f, 0] = out[f, 0] * scale + ox
    out[f, 1] = out[f, 1] * scale + oy
    return out


def unletterbox_kpts(kpts: np.ndarray, scale: float, ox: int, oy: int) -> np.ndarray:
    out = kpts.copy()
    f = out[:, 2] > 0
    out[f, 0] = (out[f, 0] - ox) / scale
    out[f, 1] = (out[f, 1] - oy) / scale
    return out


# --- rendering ----------------------------------------------------------------------------------
def render_skeleton(kpts: np.ndarray, size: tuple[int, int] = CANVAS, *, joint_radius: float = 3,
                    bone_width: int = 2) -> Image.Image:
    """(17,3) keypoints -> OpenPose-coloured skeleton on black, (W,H) = size. RINO's encode_pose with
    configurable sizes; joints are drawn after bones and must stay thicker than them for the decoder."""
    w, h = size
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    vis = np.asarray(kpts)[:, 2] > 0
    for a, b in COCO_SKELETON:
        if vis[a] and vis[b]:
            d.line([tuple(map(float, kpts[a, :2])), tuple(map(float, kpts[b, :2]))],
                   fill=tuple(int(c) for c in COCO_COLORS[a]), width=int(bone_width))
    r = float(joint_radius)
    for j in range(N_COCO):
        if vis[j]:
            x, y = float(kpts[j, 0]), float(kpts[j, 1])
            d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(c) for c in COCO_COLORS[j]))
    return img


# --- colour-free decoding (RINO codec.py) -----------------------------------------------------
def detect_peaks(gen: Image.Image, sat_thresh: float = 60.0, min_radius: float = 2.5, nms: int = 12) -> np.ndarray:
    rgb = np.asarray(gen.convert("RGB"), dtype=np.float32)
    mask = (rgb.max(-1) - rgb.min(-1)) > sat_thresh
    if mask.sum() < 10:
        return np.zeros((0, 3))
    dt = ndimage.distance_transform_edt(mask)
    mx = ndimage.maximum_filter(dt, size=nms)
    ys, xs = np.nonzero((dt == mx) & (dt >= min_radius))
    pts = np.stack([xs, ys, dt[ys, xs]], 1).astype(np.float32)
    pts = pts[np.argsort(-pts[:, 2])]
    keep = []
    for p in pts:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > nms ** 2 for q in keep):
            keep.append(p)
    return np.array(keep) if keep else np.zeros((0, 3))


def _label_head(gen: Image.Image, out: np.ndarray, center, shoulder_w: float):
    cx, cy = center
    box = max(12, int(0.45 * shoulder_w))
    rgb = np.asarray(gen.convert("RGB"), dtype=np.float32)
    H, W = rgb.shape[:2]
    x0, x1 = max(0, int(cx - box)), min(W, int(cx + box))
    y0, y1 = max(0, int(cy - box)), min(H, int(cy + box))
    sub = rgb[y0:y1, x0:x1]
    head = [(cx, cy, 1.0)] * 5
    if sub.size:
        m = (sub.max(-1) - sub.min(-1)) > 60
        if m.sum() > 6:
            dt = ndimage.distance_transform_edt(m)
            mx = ndimage.maximum_filter(dt, size=5)
            ys, xs = np.nonzero((dt == mx) & (dt >= 1.5))
            pk = sorted(zip(xs + x0, ys + y0, dt[ys, xs]), key=lambda p: -p[2])
            kept = []
            for p in pk:
                if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 36 for q in kept):
                    kept.append(p)
            if len(kept) >= 3:
                nose = max(kept, key=lambda p: p[1])
                rest = sorted([p for p in kept if p is not nose], key=lambda p: p[0])

                def shrink(p, f):
                    return (nose[0] + f * (p[0] - nose[0]), nose[1] + f * (p[1] - nose[1]), p[2])
                if len(rest) >= 4:
                    r_ear, r_eye = shrink(rest[0], .75), shrink(rest[1], .45)
                    l_eye, l_ear = shrink(rest[-2], .45), shrink(rest[-1], .75)
                    head = [nose, l_eye, r_eye, l_ear, r_ear]
                else:
                    r_eye, l_eye = shrink(rest[0], .45), shrink(rest[-1], .45)
                    head = [nose, l_eye, r_eye, l_eye, r_eye]
    for j in range(5):
        out[j] = head[j]


def decode_pose(gen: Image.Image, iters: int = 4) -> np.ndarray:
    """Generated RGB skeleton -> (17,3) keypoints (x, y, confidence) in canvas pixels, colour-free."""
    pts = detect_peaks(gen)
    out = np.zeros((N_COCO, 3), np.float32)
    if len(pts) < 4:
        return out
    xy = pts[:, :2]
    btmpl = TEMPLATE[BODY]
    mn, mx = xy.min(0), xy.max(0)
    span = np.maximum(mx - mn, 1.0)
    tmpl = btmpl * span + mn
    used = np.zeros(len(pts), bool)
    s, t = 1.0, np.zeros(2)
    for _ in range(iters):
        cost = np.linalg.norm(xy[:, None, :] - tmpl[None, :, :], axis=-1)
        ri, ci = linear_sum_assignment(cost)
        out[:] = 0
        used[:] = False
        for r, c in zip(ri, ci):
            out[BODY[c]] = (pts[r, 0], pts[r, 1], pts[r, 2])
            used[r] = True
        f = np.array([out[j, 2] > 0 for j in BODY])
        if f.sum() < 4:
            break
        src, dst = btmpl[f], out[np.array(BODY)[f], :2]
        muS, muD = src.mean(0), dst.mean(0)
        s = np.sqrt(((dst - muD) ** 2).sum() / (((src - muS) ** 2).sum() + 1e-9))
        t = muD - s * muS
        tmpl = s * btmpl + t
    nose_xy = s * TEMPLATE[0] + t
    free = np.where(~used)[0]
    if len(free):
        k = free[np.argmin(np.linalg.norm(xy[free] - nose_xy, axis=1))]
        _label_head(gen, out, (pts[k, 0], pts[k, 1]), float(s * np.linalg.norm(TEMPLATE[5] - TEMPLATE[6])))
    return out


# --- metrics (RINO metrics.py) ----------------------------------------------------------------
def _fit_scale_trans(P, G):
    muP, muG = P.mean(0), G.mean(0)
    s = np.sqrt(((G - muG) ** 2).sum() / (((P - muP) ** 2).sum() + 1e-12))
    return s, muG - s * muP


def align_similarity(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    m = (pred[:, 2] > 0) & (gt[:, 2] > 0)
    if m.sum() < 3:
        return pred.copy()
    P, G = pred[m, :2], gt[m, :2]
    s, t = _fit_scale_trans(P, G)
    for _ in range(2):
        res = np.linalg.norm(s * P + t - G, axis=1)
        keep = res <= max(2.0 * np.median(res), 1e-6)
        if keep.sum() < 3 or keep.all():
            break
        s, t = _fit_scale_trans(P[keep], G[keep])
    out = pred.copy()
    f = pred[:, 2] > 0
    out[f, :2] = s * pred[f, :2] + t
    return out


def metrics(pred: np.ndarray, gt: np.ndarray, bbox: tuple[float, float]) -> dict[str, float]:
    """Per-sample pooled counts (c05.. raw, p05.. PA) + OKS; empty if no visible GT joint."""
    vis = gt[:, 2] > 0
    n_vis = int(vis.sum())
    if n_vis == 0:
        return {}
    ref = float(max(bbox[0], bbox[1]))
    area = max(bbox[0] * bbox[1], 1.0)
    row: dict[str, float] = {"n_vis": float(n_vis)}
    for tag, p in (("c", pred), ("p", align_similarity(pred, gt))):
        found = p[:, 2] > 0
        d = np.linalg.norm(p[:, :2] - gt[:, :2], axis=1)
        d[~found] = np.inf
        for thr in PCK_THRESHOLDS:
            row[f"{tag}{int(thr * 100):02d}"] = float((d[vis] <= thr * ref).sum())
        if tag == "p":
            e = d[vis] ** 2 / (2 * area * (OKS_SIGMAS[vis] ** 2) + 1e-12)
            row["oks"] = float(np.exp(-e).mean())
    return row


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    nvis = sum(r["n_vis"] for r in rows)
    out: dict[str, float] = {"n_samples": float(len(rows)), "n_keypoints": float(nvis)}
    for thr in PCK_THRESHOLDS:
        ti = int(thr * 100)
        out[f"pck@{thr}"] = float(sum(r[f"c{ti:02d}"] for r in rows) / max(nvis, 1))
        out[f"pa_pck@{thr}"] = float(sum(r[f"p{ti:02d}"] for r in rows) / max(nvis, 1))
    oks = np.array([r["oks"] for r in rows])
    out["oks_mean"] = float(oks.mean())
    out["oks_ap"] = float(np.mean([(oks >= t).mean() for t in np.arange(0.5, 1.0, 0.05)]))
    return out


# --- colour-aware decoding (ours, not RINO) ---------------------------------------------------
def decode_pose_colour(gen: Image.Image, *, sat_thresh: float = 60.0, min_radius: float = 2.0,
                       max_colour_dist: float = 80.0) -> np.ndarray:
    """Decode by the OpenPose joint colours instead of RINO's upright template: joint discs are the
    distance-transform peaks (bones are thinner), each disc's mean colour is matched to the 17 joint
    colours by Hungarian assignment. Meant for models trained to reproduce the exact palette; RINO's
    `decode_pose` stays the default for comparability with published zero-shot numbers."""
    rgb = np.asarray(gen.convert("RGB"), dtype=np.float32)
    mask = (rgb.max(-1) - rgb.min(-1)) > sat_thresh
    out = np.zeros((N_COCO, 3), np.float32)
    if mask.sum() < 10:
        return out
    dt = ndimage.distance_transform_edt(mask)
    ys, xs = np.nonzero((dt == ndimage.maximum_filter(dt, size=5)) & (dt >= min_radius))
    if not len(xs):
        return out
    order = np.argsort(-dt[ys, xs])
    peaks = []
    for i in order:  # NMS: one peak per disc
        x, y, r = xs[i], ys[i], dt[ys[i], xs[i]]
        if all((x - px) ** 2 + (y - py) ** 2 > max(r, pr) ** 2 for px, py, pr in peaks):
            peaks.append((x, y, r))
    colours = []
    for x, y, r in peaks:
        y0, y1, x0, x1 = int(y - r + 1), int(y + r), int(x - r + 1), int(x + r)
        patch, inside = rgb[max(0, y0):y1, max(0, x0):x1], mask[max(0, y0):y1, max(0, x0):x1]
        colours.append(patch[inside].mean(0) if inside.any() else rgb[y, x])
    cost = np.linalg.norm(np.asarray(colours)[:, None, :] - COCO_COLORS[None, :, :], axis=-1)
    for r, c in zip(*linear_sum_assignment(cost)):
        if cost[r, c] <= max_colour_dist:
            out[c] = (peaks[r][0], peaks[r][1], peaks[r][2])
    return out
