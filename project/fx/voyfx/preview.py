"""Contact sheets: N frames sampled across time, labelled with timecode."""
import cv2
import numpy as np

from .io import FrameReader, probe


def contact_sheet(path, out_png, n=6, cols=6, thumb_w=360, label=None, times=None, compare=None):
    """Write a PNG grid of n frames. `compare`: second video shown as a row underneath (before/after)."""
    info = probe(path)
    if times is None:
        idx = [int(round(k * (info.nframes - 1) / max(n - 1, 1))) for k in range(n)]
    else:
        idx = [min(info.nframes - 1, int(round(float(t) * info.fps))) for t in times]
        n = len(idx)
    rows_src = [path] + ([compare] if compare else [])
    th = int(round(thumb_w * info.h / info.w))
    sheets = []
    for src in rows_src:
        want = set(idx)
        got = {}
        rd = FrameReader(src, size=(thumb_w, th))
        for i, f in enumerate(rd):
            if i in want:
                got[i] = f
            if i >= max(idx):
                break
        thumbs = []
        for i in idx:
            f = got.get(i)
            if f is None:
                f = np.zeros((th, thumb_w, 3), np.uint8)
            f = f.copy()
            txt = f"{i / info.fps:5.2f}s  f{i}"
            cv2.putText(f, txt, (8, th - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(f, txt, (8, th - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(f)
        sheets.append(thumbs)
    cols = min(cols, n)
    rows = []
    for thumbs in sheets:
        for r in range(0, n, cols):
            row = thumbs[r:r + cols]
            while len(row) < cols:
                row.append(np.zeros_like(thumbs[0]))
            rows.append(np.concatenate([np.pad(x, ((2, 2), (2, 2), (0, 0))) for x in row], 1))
    sheet = np.concatenate(rows, 0)
    head = np.full((36, sheet.shape[1], 3), 18, np.uint8)
    cv2.putText(head, label or path.split("/")[-1], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    sheet = np.concatenate([head, sheet], 0)
    cv2.imwrite(out_png, sheet)
    return out_png


def single_frame(path, out_png, t=None, frame=None, scale=1.0):
    info = probe(path)
    k = frame if frame is not None else int(round((t or 0) * info.fps))
    rd = FrameReader(path)
    for i, f in enumerate(rd):
        if i == k:
            if scale != 1.0:
                f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(out_png, f)
            return out_png
    raise IndexError(k)
