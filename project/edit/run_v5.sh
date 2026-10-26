#!/bin/bash
# v5: per-version picture (cuts/transitions on each music), mux master + light
cd /home/claude/voynich/edit
V=$1
python3 mix2.py $V > mix_v5_$V.log 2>&1 || exit 1
python3 render.py video_v5_$V.mp4 --ver $V --sheet sheet_v5_$V.png > render_v5_$V.log 2>&1 || exit 1
OUT=/mnt/user-data/outputs
ffmpeg -v error -y -i video_v5_$V.mp4 -i mix_$V.wav -map 0:v -map 1:a -c:v libx264 -preset medium -crf 21 -maxrate 16M -bufsize 32M \
       -pix_fmt yuv420p -c:a aac -b:a 256k -shortest -movflags +faststart $OUT/voynich_v5_$V.mp4
ffmpeg -v error -y -i video_v5_$V.mp4 -c:v libx264 -preset slow -b:v 1500k -pass 1 -passlogfile /tmp/claude-0/x5_$V -an -f null /dev/null
ffmpeg -v error -y -i video_v5_$V.mp4 -i mix_$V.wav -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 1500k -pass 2 -passlogfile /tmp/claude-0/x5_$V \
       -pix_fmt yuv420p -c:a aac -b:a 160k -shortest -movflags +faststart $OUT/voynich_v5_${V}_light.mp4
touch /home/claude/voynich/edit/.v5done_$V
