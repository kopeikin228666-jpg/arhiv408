# Brief for the agent team — episode «408 · Дело №001» (one video, ~60 s)

## What it is
A vertical (1080x1920, 30 fps) dark pixel-art horror short in Russian for TikTok / Reels / Shorts / VK Clips.
The host is «408»: a small hooded figure, faceless darkness, two narrow red glowing eye slits, a mouth that is a
digital sound wave when he speaks, black ragged cloak with smoke, big black boots. He lives in his lab (wall of old CRT
monitors, mainframes with tape reels, a swinging lamp, fog on the floor). He tells the case of the real Voynich
manuscript (MS 408, Beinecke Library, Yale), then dives into the big screen and the horror reel plays.
Character bible: /home/claude/voynich/CHARACTER_408.md.  The client wants it PERFECT and VIRAL ("вирус").

## Current state
- Builder: /home/claude/voynich/team/host/make_ep1d.py   (run with env PART=3 from /home/claude/voynich/edit:
  `cd /home/claude/voynich/edit && PART=3 python3 /home/claude/voynich/team/host/make_ep1d.py [--audio-only|--video-only] [--from S --to S]`)
  It writes /home/claude/voynich/team/host/ep1d_p3/ep.wav and ep_video.mp4 (+ preview jpgs f_*.jpg every second).
- Final muxed video (when the current render finishes): /mnt/user-data/outputs/408_delo001.mp4
- Edit (seconds):  hook 0–6.4 (408 close-up «Эта книга… не хочет, чтобы её читали» + flash cuts + title «ДЕЛО №001 /
  РУКОПИСЬ ВОЙНИЧА») · lab 6.4–12.6 «Двести сорок страниц… и ни одного… прочитанного слова» (kinetic 240 / 0) ·
  reel 12.6–14.3 (pages) · close 14.3–17.9 «Она написана… неизвестно кем» («АВТОР: ???») · reel 17.9–19.9 (hourglass
  «Шестьсот лет назад», 1404–1438) · lab 19.9–26.6 (he walks, turns, points) «В 1912-м её купили. Потом пытались читать» ·
  reel 26.6–29.6 «Никто… не смог» · reel 29.6–33.7 «Буквы двигаются, когда никто не смотрит» (408 live window in the corner)
  · lean 33.7–35.1 (he leans into the lens) · dive 35.1–36.6 (into the big monitor) · reel 36.6–53.4 «Растения… неправильные /
  Женщины в зелёной воде / Их тела чуть-чуть не те / Как будто художник видел то, чего нельзя видеть» · reel 53.4–58.3
  whisper «Она уже… внутри…» · card 58.3–60.7 hard cut to black «ПИШИ, ЧТО ТЫ ВИДИШЬ».
- Voice: CLEAR (the client rejected bitcrush). Reel audio comes from /home/claude/voynich/edit/mix_B_C.wav (film time).
- Code: edit/host.py (the character; sprite_img / draw), edit/room.py (the lab; frame()), edit/render.py (captions
  draw_caption, transitions run_trans, tear, rgbsplit), edit/transitions.py, look/pixelize.py (18-colour palette).

## The client's standing requirements (never break)
1. Pixel-art look, 18-colour palette, 180x320 grid upscaled x6.  2. Voice never cut off; every word clearly audible
(speech band ≥ 8 dB over music+fx).  3. Subtitles word-exact, never late.  4. Picture, voice and subtitles match in meaning.
5. Many varied transitions, motion on the beat.  6. Dark, gloomy, scary.  7. Viewers must understand it is a REAL book
(Voynich manuscript).  8. 408 moves a lot: walks, points, turns, leans, eyes react.  9. Sound balanced: nothing suddenly
too loud, nothing inaudible.  10. Viral: the first second must stop the scroll, the pace must never sag, the ending hits.

## Rules
- Only edit the files you own (named in your task). Put scratch work in /home/claude/voynich/team/agents/<your-name>/.
- The machine has 2 CPU cores shared with a render: keep tests short (single frames, a few seconds of audio). No pip
  installs; use numpy / scipy / opencv / ffmpeg that are already installed.
- No internet generation tools (no credits). No new voice lines exist: only the recordings in /home/claude/voynich/audio.
- Report in your final message (subagents cannot write .md reports): what you changed, how you measured it, what remains.
