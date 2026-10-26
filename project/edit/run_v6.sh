#!/bin/bash
cd /home/claude/voynich/edit
V=B
python3 render.py video_v6_$V.mp4 --ver $V --sheet sheet_v6_$V.png > render_v6_$V.log 2>&1 || exit 1
touch .v6video_$V
