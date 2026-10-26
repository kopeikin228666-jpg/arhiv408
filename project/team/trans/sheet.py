"""contact sheet of the new transition kinds (each on the pair of keyframes it is planned for)"""
import sys
sys.path.insert(0, '/home/claude/voynich/edit')
import numpy as np, cv2
import transitions as T

K = '/home/claude/voynich/kf/'
ROWS = [('inkbleed', 's02_parchment', 's19_water', 19),
        ('pageturn', 's28_corner', 's29_bookopen', 28),
        ('ripple', 's20_pools', 's21_woman', 20),
        ('pixelrain', 's15_reading', 's16_hourglass', 15),
        ('blink', 's03_eye', 's04_museum', 4),
        ('textdissolve', 's30_pov', 's31_lips', 31),
        ('mirror', 's10_grid', 's35_whisperer', 35)]
PROGS = (0.06, 0.18, 0.3, 0.42, 0.5, 0.58, 0.7, 0.82, 0.94)
only = sys.argv[1:] or None
rows = [r for r in ROWS if not only or r[0] in only]
gap, lab = 3, 96
sc = 1
sheet = np.zeros((len(rows) * (T.H + gap) + 20, lab + len(PROGS) * (T.W + gap), 3), np.uint8)
sheet[:] = T.C['void2']
for j, pr in enumerate(PROGS):
    cv2.putText(sheet, f'{pr:g}', (lab + j * (T.W + gap) + 76, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)
for i, (name, ka, kb, seed) in enumerate(rows):
    a, b = T._load_kf(K + ka + '.jpg'), T._load_kf(K + kb + '.jpg')
    y = 20 + i * (T.H + gap)
    cv2.putText(sheet, name, (4, y + T.H // 2), cv2.FONT_HERSHEY_PLAIN, 1.0, (220, 230, 230), 1)
    for j, pr in enumerate(PROGS):
        x = lab + j * (T.W + gap)
        sheet[y:y + T.H, x:x + T.W] = T.ALL[name](a, b, pr, seed)
out = '/home/claude/voynich/team/trans/sheet.png' if not only else '/home/claude/voynich/team/trans/sheet_' + '_'.join(only) + '.png'
cv2.imwrite(out, sheet)
print(out)
