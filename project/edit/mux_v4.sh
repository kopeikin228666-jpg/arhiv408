#!/bin/bash
# v3: three versions, master (crf 21) + light copy (<30 MB)
set -e
cd /home/claude/voynich/edit
OUT=/mnt/user-data/outputs
for V in B C; do
  ffmpeg -v error -y -i video_v4.mp4 -i mix_$V.wav -map 0:v -map 1:a -c:v libx264 -preset medium -crf 21 -maxrate 16M -bufsize 32M \
         -pix_fmt yuv420p -c:a aac -b:a 256k -shortest -movflags +faststart $OUT/voynich_v4_$V.mp4 &
done
wait
for V in B C; do
  ffmpeg -v error -y -i video_v4.mp4 -c:v libx264 -preset slow -b:v 1500k -pass 1 -passlogfile /tmp/claude-0/x4_$V -an -f null /dev/null
  ffmpeg -v error -y -i video_v4.mp4 -i mix_$V.wav -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 1500k -pass 2 -passlogfile /tmp/claude-0/x4_$V \
         -pix_fmt yuv420p -c:a aac -b:a 160k -shortest -movflags +faststart $OUT/voynich_v4_${V}_light.mp4 &
done
wait
touch /home/claude/voynich/edit/.mux4done
ls -la $OUT | grep v4
