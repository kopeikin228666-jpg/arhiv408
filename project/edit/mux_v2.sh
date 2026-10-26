#!/bin/bash
# masters (full quality) + light copies (<30 MB) for the phone / Music folder
set -e
cd /home/claude/voynich/edit
OUT=/mnt/user-data/outputs
mkdir -p $OUT
for V in A B; do
  ffmpeg -v error -y -i video_v2.mp4 -i mix_$V.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 256k -shortest -movflags +faststart $OUT/voynich_v2_$V.mp4
  ffmpeg -v error -y -i video_v2.mp4 -c:v libx264 -preset slow -b:v 1500k -pass 1 -passlogfile /tmp/claude-0/x264_$V -an -f null /dev/null
  ffmpeg -v error -y -i video_v2.mp4 -i mix_$V.wav -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 1500k -pass 2 -passlogfile /tmp/claude-0/x264_$V \
         -pix_fmt yuv420p -c:a aac -b:a 160k -shortest -movflags +faststart $OUT/voynich_v2_${V}_light.mp4
done
ls -la $OUT
