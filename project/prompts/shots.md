# Войнич — промпты для генерации (пиксельная версия)

Общие правила для всех кадров (ВСЁ ПИКСЕЛЬНОЕ):
- Начало каждого промпта картинки (PIXEL PREFIX):
  `Pixel art, true low-resolution pixel art with crisp square pixels, no anti-aliasing, no smooth gradients, limited dark palette (near-black, cold grey-greens, bone white, swamp green, faded parchment yellow, one blood-red accent), subtle ordered dithering in shadows, vertical 9:16, premium indie horror game look, no text, no UI.`
- Персонажи — сохранённые пиксельные Character в Creative Claw (Читатель / Шепчущая), передаются через character_id.
- Видеопромпт (VIDEO SUFFIX): `Keep the pixel-art look: crisp square pixels, limited palette, no smoothing, no realism. Slow, minimal motion; movement happens in small stepped pixel increments like a low-framerate sprite animation.`
- ОДОБРЕННАЯ ОСНОВА: look/pixel_reader.png. Конвейер: генерируем реалистичный (или пиксельный) кадр/клип → мой пиксель-конвертер (сетка 180×320, палитра ролика, ordered dither, ×6 до 1080×1920) → персонаж вырезается по маске и распадается на блоки с одного края, блоки отрываются и уплывают → пиксельный UI. Реалистичная генерация даёт лучшую анатомию и движение, пиксель делаем мы сами.
- Поверх — пиксельный моушен-дизайн (одобренный кадр pixel_reader.png): рамки трекинга SUBJECT_01, красные маркеры, направляющие, плитка распадается в пустоту, пиксели отрываются и уплывают, субтитры капсом с ключевым словом в красной плашке.
- Переходы: пиксельные «вайпы» блоками 8×8, dither-dissolve, datamosh, pixel-sort.

## Библия персонажей (опорные изображения — делаем первыми и утверждаем)

**READER (Читатель) — пиксельный, сохранён в Creative Claw.** Tall thin figure in a long dark grey-green hooded coat, hood up; instead of a face a blank parchment oval with one Voynich glyph and two small glowing green round lenses; pale long-fingered hands; open book with a red ribbon; coat hem dissolves into falling square pixels.

**WHISPERER (Шепчущая) — пиксельная, сохранена в Creative Claw.** Slender female figure of dark swamp-green water wrapped in torn strips of yellow parchment with ink glyphs, long black wet hair, no mouth, pale plants growing from the shoulders, water pixels dripping from the fingers.

**SILHOUETTE (Тень у витрины).** Plain dark human silhouette in a long coat, only visible as a reflection in glass. Делается в композитинге, генерировать не нужно.

**Места:** MUSEUM — dark empty museum hall, single glass vitrine on a pedestal, one cold overhead light. READING ROOM — empty old library reading room, one green banker's lamp, wooden chair. POOLS — interconnected stone bathing pools with organic tubes, murky green water, mist. BEDROOM — dark room, bed shaped like a giant open manuscript page.

## Кадры

Формат: **N · время · тип** — `IMG` ключевой кадр (картинка), `VID` движение (image-to-video от ключевого кадра), `POST` делается в монтаже без генерации.

### Акт I

**1 · 0:00–0:01 · POST** — чёрный, разрыв, глиф. Глиф берём из скана рукописи.

**2 · 0:01–0:02.5**
- IMG: Extreme macro of aged vellum parchment, visible fibers and cracks, faded brown ink of unreadable looping script, raking side light.
- VID: Very slow push-in, subtle handheld breathing; ink letters almost imperceptibly shift.

**3 · 0:02.5–0:03.5**
- IMG: Extreme macro of a human eye (READER), instead of a pupil a tiny dark ink-drawn plant with roots spreading across the iris like cracks, wet skin texture.
- VID: Static, the eye slowly blinks once at the very end.

**4 · 0:03.5–0:05** (MUSEUM)
- IMG: Open ancient manuscript inside a glass vitrine in the dark empty museum hall, one cold overhead light, the glass reflects the empty hall.
- VID: Slow dolly-in toward the vitrine; a man (READER) slowly walks past behind the vitrine, left to right.
- Примечание: отражение без человека делаем в посте (echo / маска).

**5 · 0:05–0:06 · POST** — строб страниц из сканов рукописи.

**6 · 0:06–0:08**
- IMG: Side view, a pale hand (READER's, dark coat sleeve) slowly reaching toward an open vellum page with botanical drawings, dim light.
- VID: The hand slowly reaches and fingertips touch the page; hold.

**7 · 0:08–0:09 · POST** — титр «НЕ ЧИТАЙ».

**8 · 0:09–0:11**
- IMG: Old archival photograph, sepia, a man in a 1910s overcoat and hat standing over an open wooden crate of old books in a dim villa hallway; his face is out of focus and in shadow.
- VID: Slow Ken Burns push-in (можно сделать в посте из картинки).

**9 · 0:11–0:12**
- IMG: Close-up of an old paper auction tag with a handwritten lot number tied with string to a leather-bound book, dark background.
- VID: Static, the tag sways slightly.

**10 · 0:12–0:14** — 7 портретов + чёрный экран (8-й делаем в посте)
- IMG ×7, each: Head-and-shoulders portrait of a [VARIANT] staring intensely at something below camera, dim lamp light, dark background.
  - VARIANTS: 1920s cryptographer with slicked hair and waistcoat · 1940s military codebreaker in uniform · elderly scholar holding a magnifying glass · woman librarian in 1950s cardigan · 1970s academic with beard and turtleneck · young researcher lit by a laptop screen · READER.
- VID: Each slowly raises eyes to look straight into the camera.

**11 · 0:14–0:16**
- IMG: A decipherer's wall covered with handwritten frequency tables, papers, photos of manuscript pages, red strings connecting symbols, a single desk lamp.
- VID: Fast whip-pan left across the wall (скорость доделаем в посте).

**12 · 0:16–0:18**
- IMG: Extreme macro diving into a vellum page: ink strokes, then plant roots drawn in ink becoming real roots.
- VID: Continuous fast push-in forward through the page into darkness.

**13 · 0:18–0:19.5 · POST** — сетка 3×3 из клипов кадра 10.

**14 · 0:19.5–0:20 · POST** — чёрный, tape-stop.

### Акт II

**15 · 0:20–0:23** (READING ROOM)
- IMG: Empty old library reading room at night, one green banker's lamp on a long wooden table, an empty wooden chair, deep shadows.
- VID: Very slow push-in; the empty chair rocks almost imperceptibly by itself.

**16 · 0:23–0:28**
- IMG: An antique hourglass standing on old vellum in darkness, dark sand.
- VID: Slow 30-degree orbit around the hourglass. (Песок вверх — реверс в посте.)

**17 · 0:28–0:32**
- IMG: Profile close-up of READER in dim light, lips slightly parted, cold air visible.
- VID: He slowly moves his lips silently, faint breath vapor drifts out of his mouth. (Глифы вместо пара — в посте.)

**18 · 0:32–0:36**
- IMG: Locked-off close-up of one line of handwritten Voynich-like script on vellum, flat soft light.
- VID: Completely static — использовать картинку, перестановка строк делается в посте.

**19 · 0:36–0:40**
- IMG: Top-down view of dark still green water; pale roots slowly rising from below; on the surface, the reflection of a strange plant with a leaf shaped like an opening hand.
- VID: Very slow clockwise rotation; roots slowly curl like fingers.

**20 · 0:40–0:45** (POOLS)
- IMG: High overhead view of the stone bathing pools; pale women in wet linen stand waist-deep in murky green water, perfectly still, looking up.
- VID: Slow crane descent toward the pools; the figures do not move.

**21 · 0:45–0:50**
- IMG: Medium shot of one pale woman in wet linen standing waist-deep in green water among others, mist.
- VID: All the women slowly raise their arms at the same time. (Отставание одной фигуры и обратный локоть — в посте.)

**22 · 0:50–0:54**
- IMG: Underwater shot looking up through murky green water at the surface; dark silhouettes of women standing above, all looking down.
- VID: Camera slowly sinks deeper, bubbles rise.

**23 · 0:54–0:58**
- IMG: Over-the-shoulder top view of a medieval scribe's hand with a quill drawing on vellum by candlelight; the drawing is a portrait of READER's face in ink.
- VID: The hand slowly continues drawing, candle flickers.

**24 · 0:58–0:59 · POST** — зелёная вода заливает экран.

**25 · 0:59–1:04** (MUSEUM)
- IMG: Close-up of the glass vitrine with the closed manuscript; the glass is fogging from the inside.
- VID: Slow push-in; condensation spreads on the inside of the glass and a handprint appears on it from the inside.

**26 · 1:04–1:08**
- IMG: Strict profile of READER, face almost touching the vitrine glass, eyes closed.
- VID: He slowly inhales deeply; leans a few millimeters closer.

**27 · 1:08–1:12** (BEDROOM)
- IMG: Top-down view of READER sleeping on a bed that is a giant open manuscript page with faded script and plant drawings, dark room.
- VID: Slow 90-degree rotation of the camera above him; he twitches once in his sleep.

**28 · 1:12–1:15 · POST** — повторы кадра 27 + Шепчущая в углу (композит, нужен её клип ниже, №35-B).

### Акт III

**29 · 1:15–1:20**
- IMG: Closed leather-bound manuscript on an empty table in total darkness, eye-level.
- VID: The cover slowly lifts by itself, murky green light spills out from inside, pages begin to turn backwards.

**30 · 1:20–1:25**
- IMG: POV from inside a book looking out through the gap between two vellum pages; above, READER leans over, looking down into the camera, his eyes scanning lines.
- VID: Static; his eyes move left to right as if reading.

**31 · 1:25–1:30**
- IMG: Pure black frame with one thin vertical sliver of light between two pages; inside the sliver, pale lips.
- VID: The lips slowly whisper.

**32 · 1:30–1:31 · POST** — зерно и гул.

**33 · 1:31–1:38**
- IMG: Tight close-up of READER looking straight into the lens, frightened, dim cold light.
- VID: Very slow push-in to an extreme close-up of the eyes; his lips move as if speaking slowly. (Рассинхрон и распад — в посте.)

**34 · 1:38–1:41 · POST** — сетка 3×3 с Читателем в центре.

**35 · 1:41–1:48**
- IMG A: WHISPERER full figure standing in darkness, front view; in her chest an open manuscript glows faintly.
- VID A: Slow push-in toward her chest; her head tilts slightly.
- IMG B (для кадра 28): WHISPERER standing still in a dark corner of a room, wide, very dim.
- VID B: Static, she does not move at all for 4 seconds.
- Рекурсия Дросте — в посте.

**36 · 1:48–1:52 · POST** — схлопывание рекурсии.

**37 · 1:52–1:58**
- IMG: Same MUSEUM vitrine shot as shot 4, now the manuscript is closed. Clean neutral light. Totally still.
- VID: Не нужно — статичная картинка. Силуэт в отражении — композит.

**38 · 1:58–2:00**
- VID: Из кадра 37: the manuscript's cover lifts by one centimeter and falls closed. Титр и петля — в посте.

## Голоса (для ElevenLabs / Voice Design)

- **Нарратор:** Very low, hoarse, slow male voice, almost a whisper, close to the microphone, tired, uncertain, Russian. Реплики — из сценария, паузы длинные.
- **Читатель (Герой 1):** Male, mid-30s, quiet, frightened, breathy, Russian. «Ты слишком долго смотришь…»
- **Шепчущая (Герой 2):** Female, airy, extremely quiet whisper, wet breathy texture, Russian. «Она уже… внутри…»
- Все искажения (питч, bit-crush, реверс-эхо) накладываем в посте.

## Звуковые эффекты (генерация SFX)

Low sub drone 35 Hz rumble · digital glitch clicks and bit-crush bursts · mechanical relay clicks · slow turning of vellum pages (будет замедлено и развёрнуто) · quill scratching on parchment, very close · underwater pressure and drops · slow long human inhale against glass · many overlapping whispers of nonsense syllables · tape stop.
