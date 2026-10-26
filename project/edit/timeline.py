"""Shared placement of every voice line on the film timeline (v2: 30 s intro, total 130 s).
Each line is played WHOLE, never split. Keywords marked with * go on the red plate.
Spoken forms in {} replace the written word for syllable counting.
"""
DUR = 130.0
SHIFT = 10.0            # old v1 timeline (from 0:00 act I) -> new film time

A = '/home/claude/voynich/audio'

# (file, src_start, src_end, film_start, text, voice)
LINES = [
    ('vo_intro.mp3', 0.00, 5.49, 1.20, 'Двести сорок *страниц… и ни одного… *прочитанного слова.', 'intro'),
    ('vo_narrator.mp3', 0.00, 3.81, 13.60, 'Эта *книга… не хочет, чтобы её *читали.', 'narr'),
    ('vo_narrator.mp3', 4.74, 8.07, 19.20, 'В 1912-м{тысяча девятьсот двенадцатом} её *купили.', 'narr'),
    ('vo_narrator.mp3', 8.51, 10.76, 22.75, 'Потом пытались *читать.', 'narr'),
    ('vo_narrator.mp3', 11.27, 13.03, 27.70, '*Никто… не *смог.', 'nikto'),
    ('vo_written.mp3', 0.06, 5.44, 33.08, 'Она *написана… *неизвестно *кем. Шестьсот лет *назад.', 'narr'),
    ('vo_narrator.mp3', 21.07, 22.80, 40.40, 'Язык… *ничей.', 'narr'),
    ('vo_narrator.mp3', 23.37, 26.67, 42.80, 'Буквы *двигаются, когда никто не *смотрит.', 'narr'),
    ('vo_narrator.mp3', 27.11, 29.31, 47.60, 'Растения… *неправильные.', 'narr'),
    ('vo_narrator.mp3', 29.78, 31.99, 51.80, 'Женщины в *зелёной воде.', 'narr'),
    ('vo_narrator.mp3', 32.58, 35.10, 56.40, 'Их тела чуть-чуть{чуть чуть}… *не *те.', 'narr'),
    ('vo_narrator.mp3', 35.45, 38.58, 59.27, 'Как будто *художник видел то, чего *нельзя видеть.', 'narr'),
    ('vo_narrator.mp3', 39.18, 42.91, 69.40, 'Некоторые страницы *пахнут. Даже через *стекло.', 'narr'),
    ('vo_narrator.mp3', 43.49, 46.39, 78.00, 'После них снятся одни и те же *сны.', 'narr'),
    ('vo_narrator.mp3', 46.99, 49.94, 85.60, 'В этих снах книга *открывается *сама.', 'narr'),
    ('vo_narrator.mp3', 50.49, 52.77, 89.10, 'И кто-то{кто то} *шепчет с другой стороны.', 'narr'),
    ('vo_reader.mp3', 0.00, 2.68, 105.00, 'Ты слишком долго *смотришь…', 'reader'),
    ('vo_whisper.mp3', 0.00, 1.12, 110.60, 'Она уже…', 'whisper'),
    ('vo_whisper.mp3', 1.76, 2.51, 112.40, '*внутри…', 'whisper'),
    ('vo_narrator.mp3', 53.31, 56.35, 121.20, 'Может быть… она никогда и не была *книгой.', 'final'),
]

# Version A: track1 (аллея кошмаров) intro, creaky transition into Mara.  Version B: Mara only.
MARA_BOOM = 94.27        # Mara song time that lands on film 30.0 (act II)
ACT2 = 30.0
T1_FILM0 = 5.3           # film time where track1 song 0.0 starts (its drop ~5.1 lands on ~10.4)
B_MARA_OFF = 6.66        # version B: Mara song = film + 6.66 until the transition (wall starts on 10.5)
TRANS0 = 26.0            # creaky transition window 26..30

# knocks for the act I -> act II transition (on the cut into the page dive, a second one, and the 0:30 boom)
KNOCKS = [26.34, 26.62, 30.0]
UNDERWATER = (63.5, 68.2)
