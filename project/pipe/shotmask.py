"""Per-frame figure masks for a locked-off shot, from hand-drawn silhouette polygons + GrabCut refinement.
Usage: python3 shotmask.py IN.mp4 polys.json OUT_masks.npy   (masks at 540x968, uint8, one per source frame)
polys.json: {"polys": [[[x,y],...], ...]}  in 540x968 coordinates; union = probable FG, intersection core = sure FG.
"""
import sys, json, numpy as np, cv2

W, H = 540, 968

def main(inp, pj, out):
    polys = json.load(open(pj))['polys']
    masks = []
    for p in polys:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.array(p, np.int32)], 1)
        masks.append(m)
    union = np.max(masks, 0); inter = np.min(masks, 0)
    pr = cv2.dilate(union, np.ones((13, 13), np.uint8))
    sure = cv2.erode(inter, np.ones((17, 17), np.uint8))
    ys, xs = np.where(pr)
    x0, x1, y0, y1 = max(0, xs.min() - 20), min(W, xs.max() + 20), max(0, ys.min() - 20), min(H, ys.max() + 20)
    cap = cv2.VideoCapture(inp)
    res = []
    prev = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        crop = f[y0:y1, x0:x1]
        gc = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
        gc[pr[y0:y1, x0:x1] > 0] = cv2.GC_PR_BGD
        gc[union[y0:y1, x0:x1] > 0] = cv2.GC_PR_FGD
        if prev is not None:
            gc[(prev[y0:y1, x0:x1] > 0) & (pr[y0:y1, x0:x1] > 0)] = cv2.GC_PR_FGD
        gc[sure[y0:y1, x0:x1] > 0] = cv2.GC_FGD
        bgd = np.zeros((1, 65)); fgd = np.zeros((1, 65))
        cv2.grabCut(crop, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        mk = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        full = np.zeros((H, W), np.uint8); full[y0:y1, x0:x1] = mk
        n, lab, st, _ = cv2.connectedComponentsWithStats(full)
        if n > 1:
            k = 1 + int(np.argmax(st[1:, 4])); full = (lab == k).astype(np.uint8)
        full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        prev = full
        res.append(full)
    np.save(out, np.stack(res).astype(np.uint8))
    print('masks', len(res))

if __name__ == '__main__':
    main(*sys.argv[1:4])
