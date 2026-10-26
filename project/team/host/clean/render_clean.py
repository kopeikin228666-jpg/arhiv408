"""the v6 B reel without its burned-in captions: the episode draws every caption itself, above the platform UI"""
import os, sys
os.environ['VOY_VER'] = 'B'
sys.argv = [sys.argv[0], '--ver', 'B']
sys.path.insert(0, '/home/claude/voynich/edit')
import render as R
R.draw_caption = lambda rgb, cap, t: rgb
R.render('/home/claude/voynich/edit/video_v6_B_clean.mp4', 0.0, 130.0)
print('clean ok')
