# Brief v2 — «408 · Дело 001», round 2 ("до идеала")

Read first: /home/claude/voynich/AGENTS_BRIEF_408.md (what it is, standing requirements, rules) and
/home/claude/voynich/CHARACTER_408.md (the host). This file replaces the "Current state" section there.

## Current state (all director round-1 fixes are in)
- Builder: /home/claude/voynich/team/host/make_ep1e.py  (PART=3). Run from /home/claude/voynich/edit:
  `cd /home/claude/voynich/edit && PART=3 python3 /home/claude/voynich/team/host/make_ep1e.py [--audio-only|--video-only] [--from S --to S]`
  Output folder: /home/claude/voynich/team/host/ep1e_p3/ (ep.wav, ep_video.mp4, f_*.jpg). A full video render takes ~8 min
  of both CPU cores: DO NOT run a full render. Test single frames (see ep1e_p3/test/frames.py) or --from/--to of 1–3 s.
- Delivered: /mnt/user-data/outputs/408_delo001_final.mp4 (master) and _light.mp4. 53.0 s, -14.1 LUFS, -1.2 dBFS peak.
- Reel frames now come from /home/claude/voynich/edit/video_v6_B_clean.mp4 (the v6 B reel WITHOUT burned-in captions);
  every caption (his lines and the reel's lines via R.caption_at(film_t)) is drawn by text_layer, R.CAP_BOTTOM = 96 (above
  the TikTok/Reels UI). Reel audio: /home/claude/voynich/edit/mix_B_C.wav (film time).
- Mastering: /home/claude/voynich/team/host/ep_level.py level() (equal dialogue loudness per segment, -14 LUFS, -1.3 dBTP).
- Edit (s): hook 0–6.4 · lab «Двести сорок страниц…» 6.4–9.0 · close «и ни одного… прочитанного слова» 9.0–12.3 · reel pages
  12.3–13.2 · close «Она написана… неизвестно кем» 13.2–16.8 · reel hourglass 16.8–18.8 · lab walk «В 1912-м её купили»
  18.8–22.5 · reel «Потом пытались читать» 22.5–25.1 · reel «Никто… не смог» 25.1–27.3 · reel «Буквы двигаются…» (pip)
  27.3–31.3 · lean 31.3–32.3 · dive 32.3–33.5 · reels: plants 33.5–36.1, women 36.1–38.8, bodies/artist 38.8–45.2, whisper
  «Она уже… внутри…» 45.2–48.4, museum «Может быть… она никогда и не была книгой» 48.4–51.7 · card «ТЫ ДОЧИТАЛ. / ОНА — НЕТ.»
  51.7–53.0 (hard cut, loops to frame 0 where the red eyes already glow).
- Known open issues: the word «в» in «Женщины в зелёной воде» (film 52.80–53.05) is masked by the reel bed in mix_B_C.wav;
  26 s «НЕ СМОГ.» sits on near-black; the lean (31.3–32.3) is very dark.

## File ownership in round 2 (edit ONLY what you own; use the Edit tool for small replacements, never rewrite a whole shared file)
- Editor: make_ep1e.py everything ABOVE the line `# ------------------------------------------------------------------ audio`
  (segment list, seg_frame, hook, inserts, text_layer, slam/kin, plate), plus edit/render.py transitions (run_trans & co).
- Sound: make_ep1e.py everything BELOW that audio marker (render_audio and helpers), team/host/ep_level.py,
  edit/mix2.py and the reel mixes edit/mix_B_C.wav (keep a backup before you overwrite it).
- Animator: edit/host.py and edit/room.py only.
- Director: read-only on everything; writes only in its scratch folder.
If you need a change in a file you do not own, put it in your final report as an exact patch; the lead integrates it.
Before editing, back up the file you own (cp X X.bak_r2). Scratch: /home/claude/voynich/team/agents/<name>_r2/.
The lead will run the final full render after you all report. Keep every test short — 2 CPU cores are shared by 4 agents.
