# Voynich reel — brief for the agent team (version B only)

## What it is
A 130 s vertical (1080x1920, 30 fps) dark art-house horror reel about the Voynich manuscript, in Russian, for TikTok/Reels.
Pixel-art look: everything is drawn on a 180x320 grid (palette in /home/claude/voynich/look/pixelize.py, 18 colours), upscaled x6.
Music in version B: The Haxan Cloak — "Mara" (dark ambient: swells, booms every 8.57 s in act II, a loud wall 90–116 s).
The client chose version B. Goal: a "million-views" reel — unforgettable, genuinely frightening, "everyone in shock", polished to perfection.
Latest render of B: /mnt/user-data/outputs/voynich_v5_B.mp4 (watch frames with ffmpeg; listen via spectrograms).

## Everything the client has asked for (never break any of these)
1. Pixel-art style throughout, tracking boxes / HUD / captions with the key word on a red plate.
2. Narration must never break off, never get clipped, no fragments of other phrases; whole phrases.
3. Subtitles exactly in sync with the voice, word by word; a chunk appears 0.2 s early, each word lights up when it is heard.
   Subtitles must never lag. No word may be masked: every word ≥ +8 dB over music+fx in the 300–3400 Hz band.
4. Narrator voice natural (NO pitch shift). The two hero voices («Ты слишком долго смотришь…», «Она уже… внутри…») stay distorted.
5. Picture, voice and subtitles must match in meaning (each line sits on its picture).
6. No glitch "clicks" — percussion is crushed, creaky, pixel-style kicks/knocks.
7. Many varied motion transitions (pixels growing, spreading, swirling…), never monotonous, all in time with the music.
8. All motion in time with the music; harmonious.
9. Darker, scarier, more insane — sound and picture.
10. Final title «ты дочитал. / она — нет.»
11. NEW, top priority: viewers must understand it is a REAL book — the Voynich manuscript («Рукопись Войнича»). The film never says
    its name; it feels like a made-up story. Facts you may use: Voynich manuscript, ~240 vellum pages, Beinecke Rare Book & Manuscript
    Library, Yale University (USA), call number MS 408; vellum radiocarbon-dated 1404–1438; bought in 1912 by the antiquarian
    bookseller Wilfrid Voynich (Уилфрид Войнич); written in an unknown script, never deciphered; illustrations of plants that don't exist,
    women in green pools, star charts. No credits for new narration — the name must come through on-screen text/graphics.
12. NEW: more animation INSIDE the shots — objects that move: sand falling in the hourglass, candle flames, ripples on the pools,
    leaves swaying, bubbles, dust in light beams, a lamp that flickers, a page that turns by itself, breath fog on glass, drips,
    a sleeper breathing, the tag swinging, etc. — all in time with the music.

## Code (all in /home/claude/voynich/edit)
- timeline.py — voice LINES (file, src_start, src_end, film_start, text, voice). align.py → words.json (per-word film times).
- sync.py — musical grid of version B (env VOY_VER=B): grid(), strong(), beat_events(), PULSE, DROP, book timings.
- render.py — build() returns shots (name, t0, t1, trans); render loop draws each frame on the grid; captions; beat_pulse.
  Run a preview: `cd /home/claude/voynich/edit && python3 render.py /home/claude/voynich/team/<you>/clip.mp4 --ver B --from 28 --to 31`
  (≈10 s of CPU per 1 s of film, the machine has only 2 cores shared by 5 agents — keep previews SHORT, ≤ 3 s, and prefer calling
  your own functions directly on single frames).
- book.py — the 0–10.48 s intro (pixels assemble into the book, cover opens, page flips, dive). transitions.py — transition library
  (fn(a, b, prog, seed) -> grid). trans_sfx.py — transition sounds (sfx(kind, dur, center, seed)).
- mix2.py — the audio: `python3 mix2.py B` → mix_B.wav (48 kHz), voice stem vo_B.npy, music stem bed_B.npy (16 kHz mono). ~1 min.
- Shot names in order (B): book s1 s2 … s13 s15 s16 s17 s18 s19 s20 v21 s23 s22 s24 s26 s25 s27 s28 s29 s31 s30 s31b s32 s33 s34 s35
  s36 s37 s38. Get exact times: `VOY_VER=B python3 -c "import render; [print(s.name, round(s.t0,3), round(s.t1,3), s.trans) for s in render.build()]"`
- Keyframes (source images): /home/claude/voynich/kf/*.jpg; clips: /home/claude/voynich/clips/.
- Audio sources: /home/claude/voynich/audio (track2.mp3 = Mara, vo_*.mp3 voices). No internet; NO credits left for AI voice or
  image generation — work only with what exists (you may process/re-edit existing audio and images procedurally).

## Plug-in points (each agent owns ONLY its own file(s); never edit render.py, mix2.py, sync.py, timeline.py, align.py, book.py)
- edit/hook.py (Hook & identity agent): END = seconds (may be 130 if you also place identity graphics later in the film);
                                frame(t, g, rng) -> grid or None for t < END (overlay/replacement), applied after vfx-less beat pulse,
                                before HUD and captions;
                                optional design(ver, shots, words, sync, SR, N) -> (N,2) float32 audio layer.
- edit/anim.py (Animation agent): animate(g, shot_name, tl, u, t, rng) -> grid, applied to every shot's own frame right after the shot
                                function (so transitions see it too). tl = time since the shot start, u = 0..1 progress, t = film time.
                                Fast (< 6 ms average), deterministic, palette colours only.
- edit/vfx.py (VFX agent):      post(g, t, shot_name, tl, u, rng) -> grid. Called on every frame after transitions and beat pulse,
                                before HUD and captions. Must be fast (< 8 ms per call on average) and deterministic.
- edit/trans_plan.py (Transitions agent): plan(ver, shots) -> {shot_name: (kind, frames, frames_before_cut)}; overrides the default table
                                in render.build(). May add new kinds to edit/transitions.py (you own transitions.py now; keep its API and
                                existing kinds working). Kinds handled inside render: tear blocks melt grow swirl scatter; others from
                                transitions.ALL.
- edit/sound_design.py (Sound agent): design(ver, shots, words, sync, SR, N) -> (N,2) float32 extra layer added to the fx bus;
                                optional duck(ver, shots, words, sync, SR, N) -> (N,) gain on the music bed. You may also own
                                trans_sfx.py. Final mix goes through tanh limiter; keep layer peaks sane.
- QA/director agent: owns /home/claude/voynich/team/qa/ only; reviews, measures, reports.
Put scratch files in /home/claude/voynich/team/<your-name>/.
Subagents cannot write .md report files: put your report in your final message.
