# 408 stock phrases (Callum), cut and ready

The cuts are **raw**: no "from the darkness" processing yet. At mix time, run every cut through `make_ep1e.VOICE`
(also copied into `edit/delo003/make_delo003_v2.py`), then match the lines' levels. Files are 48 kHz / 24-bit WAV with
10 ms fade-in and 40 ms fade-out. Each cut adds 50 ms before and 150 ms after the speech span, but never more than half
of the neighbouring silence. To rebuild: `python3 cut_phrases.py`.

Source: `audio/vo_vidiq_phrases.mp3` (55.48 s). I had no ASR. Speech spans come from `silencedetect -40 dB d=0.3`,
which gives 24 spans. I matched each span to its text by three checks:
1. **Order:** the spans follow the script order (stock phrases, then Дело 010 A0–A10 from `team/scripts/delo010_jwst.md` §3).
2. **Length:** each span's length fits its text. Lines with a sentence break («Пустота… ответила» and others) split into two spans.
3. **Where the sibilants fall:** a 4–10 kHz energy profile per 20 ms marks the с/ш/щ/ч/з/ф sounds, and these must fall
   where the text puts them. Examples: «Срочно в архив» shows С at the start, ч in the middle and a devoiced «-в» [ф] at
   the end. «Открываю дело» shows no sibilant at all. «Дело закрыто» has a voiced з bridging «дело→закрыто», where
   «Дело открыто» has a vowel onset after a short gap.

All 24 spans line up with the script with no leftovers and no missing lines.

| File | Source span (s) | Text | Confidence |
|---|---|---|---|
| 408_01_ya_chetyresta_vosem.wav | 0.000–1.956 | Я — четыреста восемь. | high (ч / ст / с sibilant pattern; 1st in script) |
| 408_02_otkryvayu_delo.wav | 2.483–3.844 | Открываю дело. | high (no sibilants; ends at 3.84 like the team note «Открываю дело 2.56–3.84») |
| 408_03_delo_otkryto.wav | 4.654–6.092 | Дело открыто. | medium-high (script order + vowel onset after «дело»; same length as 04, so only order and the з-test separate them) |
| 408_04_delo_zakryto.wav | 6.784–8.203 | Дело закрыто. | medium-high (voiced з fricative right after «дело») |
| 408_05_srochno_v_arhiv.wav | 8.932–10.526 | Срочно в архив. | high |
| 408_06_sleduyushchee_delo_zavtra.wav | 11.346–13.564 | Следующее дело — завтра. | high (С…щ…з pattern, longest stock line) |
| d010_A0a_uebb_smotrel_v_pustotu.wav | 14.500–16.210 | Уэбб смотрел в пустоту. | high |
| d010_A0b_pustota_otvetila.wav | 17.008–18.945 | Пустота… ответила. | high |
| d010_A1a_glubokoe_pole_habbla.wav | 19.527–21.203 | Глубокое поле Хаббла. | high (no sibilants, as expected) |
| d010_A1b_desyat_tysyach_galaktik.wav | 21.608–23.498 | Десять тысяч галактик. | high |
| d010_A2_i_odna_pustaya_tochka.wav | 23.983–25.603 | И одна пустая точка. | high |
| d010_A3a_vsyo_izvestnoe_zakryli.wav | 26.290–27.728 | Всё известное закрыли. | high |
| d010_A3b_ostalos_sem_signalov.wav | 28.116–29.805 | Осталось семь сигналов. | high |
| d010_A4a_shest_galaktiki.wav | 30.145–31.577 | Шесть — галактики. | high |
| d010_A4b_sedmoy_nichey.wav | 31.889–33.236 | Седьмой — ничей. | high |
| d010_A5_odna_liniya_sveta_bolshe_nichego.wav | 33.906–36.289 | Одна линия света. Больше — ничего. | high (one span, an inner pause < 0.3 s) |
| d010_A6a_avgust_est.wav | 36.996–38.245 | Август — есть. | medium-high |
| d010_A6b_noyabr_drugoy_ugol_est.wav | 38.933–41.251 | Ноябрь, другой угол — есть. | high |
| d010_A7_na_snimkah_habbla_i_uebba_pusto.wav | 42.004–44.846 | На снимках Хаббла и Уэбба — пусто. | high |
| d010_A8_godom_ranshe_ne_bylo_vidno.wav | 45.496–47.532 | Годом раньше его там не было видно. | high |
| d010_A9a_rannyaya_galaktika.wav | 48.155–49.452 | Ранняя галактика? | medium-high |
| d010_A9b_sverhnovaya.wav | 49.933–50.872 | Сверхновая? | high |
| d010_A9c_zvezdu_rvyot_chyornaya_dyra.wav | 51.489–53.395 | Звезду рвёт чёрная дыра? | high |
| d010_A10_smotrim_snova.wav | 54.094–55.036 | Смотрим снова. | high |
| 408_n1_nikto_ne_smog.wav | `vo_narrator.mp3` 11.27–13.03 | Никто… не смог. | certain (ElevenLabs Scribe transcript `scribe_vo_narrator.txt`; Дело 001 line) |

Notes
- `scribe_vo_408.txt` («Я 0–0.32 … дело. 3.2–3.84») belongs to `audio/vo_408.mp3`. That file is **not on disk**: it was
  only ever at the creativeclaw CDN link in `CHARACTER_408.md`, and the proxy blocks that host. make_ep1e writes the
  transcript as a hard-coded fallback. Spans 01+02 above say the same words, and their span ends at 3.84 like the old
  file did. The inner word timings differ, though (here «Я» runs to about 0.5 s), so this is a **new take** of the same
  line, not the same file.
- Mind the gap in «Открываю дело.» to «Я — 408.»: the take has a 0.53 s pause. Дело 003 v2 tightens it to 0.36 s.
- Used in: Дело 003 v2 (`edit/delo003/make_delo003_v2.py`): 01 + 02 under the hook, n1 on «ПОЧЕМУ?».
