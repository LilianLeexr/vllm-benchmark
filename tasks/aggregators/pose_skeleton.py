"""Micro-averaged PCK (raw and PA) and OKS for both decoders, as RINO's metrics.average.

A person whose generation or scoring failed counts as all visible joints missed and OKS 0
(RINO has no failed samples; this keeps the denominator fixed at all evaluated persons).
"""

from __future__ import annotations

import numpy as np

DECODERS = ("rino", "colour")
THRESHOLDS = (("05", 0.05), ("10", 0.1), ("20", 0.2))


def aggregate(records: list[dict], config: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    failed = 0
    for decoder in DECODERS:
        n_vis, correct, oks = 0.0, {f"{t}{s}": 0.0 for t in "cp" for s, _ in THRESHOLDS}, []
        failed = 0
        for record in records:
            m = record["metrics"]
            if f"{decoder}_oks" in m:
                n_vis += m["n_vis"]
                for key in correct:
                    correct[key] += m[f"{decoder}_{key}"]
                oks.append(m[f"{decoder}_oks"])
            else:
                failed += 1
                keypoints = record["sample"].load().ground_truth["keypoints"]
                n_vis += float(sum(k[2] > 0 for k in keypoints))
                oks.append(0.0)
        for s, thr in THRESHOLDS:
            out[f"{decoder}/pck@{thr}"] = correct[f"c{s}"] / max(n_vis, 1)
            out[f"{decoder}/pa_pck@{thr}"] = correct[f"p{s}"] / max(n_vis, 1)
        values = np.asarray(oks)
        out[f"{decoder}/oks_mean"] = float(values.mean()) if len(values) else 0.0
        out[f"{decoder}/oks_ap"] = float(np.mean([(values >= t).mean() for t in np.arange(0.5, 1.0, 0.05)])) if len(values) else 0.0
    out["n_samples"] = float(len(records))
    out["samples_failed"] = float(failed)
    return out
