import sys, cv2, numpy as np, glob, os
# usage: sheet.py out.jpg cols label_fn_mode files...
out = sys.argv[1]; cols = int(sys.argv[2]); files = sys.argv[3:]
ims = []
for f in files:
    im = cv2.imread(f)
    if im is None: continue
    im = cv2.resize(im, (216, 384), interpolation=cv2.INTER_AREA)
    lab = os.path.splitext(os.path.basename(f))[0]
    if lab.startswith('f_') and len(lab) == 7:  # preview f_00030 -> seconds
        lab = f"{int(lab[2:]) / 30:.1f}s"
    elif lab.startswith('t'):
        lab = lab[1:] + 's'
    cv2.rectangle(im, (0, 0), (110, 22), (0, 0, 0), -1)
    cv2.putText(im, lab, (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    # safe-zone guides: right 120px column and bottom 380px (scaled by 216/1080 = 0.2)
    cv2.line(im, (216 - 24, 0), (216 - 24, 384), (255, 0, 255), 1)
    cv2.line(im, (0, 384 - 76), (216, 384 - 76), (255, 0, 255), 1)
    ims.append(im)
rows = []
for i in range(0, len(ims), cols):
    r = ims[i:i + cols]
    while len(r) < cols: r.append(np.zeros_like(ims[0]))
    rows.append(np.hstack(r))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
