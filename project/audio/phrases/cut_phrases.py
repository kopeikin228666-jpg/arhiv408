"""Cut the 408 stock phrases (Callum, raw/unprocessed) out of vo_vidiq_phrases.mp3 (+ the known clean narrator cut).
Speech spans from ffmpeg silencedetect (-40 dB, d=0.3); each cut gets 50 ms before and 150 ms after (never past the
middle of the neighbouring silence), 10 ms fade-in / 40 ms fade-out.  48 kHz 24-bit WAV, source channels kept.
Apply the "from the darkness" chain (make_ep1e.VOICE) at mix time, not here.
usage: python3 cut_phrases.py"""
import subprocess, os
A = '/home/claude/voynich/audio'; OUT = f'{A}/phrases'
SRC = f'{A}/vo_vidiq_phrases.mp3'
SEGS = [  # (name, start, end, text)
    ('408_01_ya_chetyresta_vosem', 0.00, 1.956, 'Я — четыреста восемь.'),
    ('408_02_otkryvayu_delo', 2.483, 3.844, 'Открываю дело.'),
    ('408_03_delo_otkryto', 4.654, 6.092, 'Дело открыто.'),
    ('408_04_delo_zakryto', 6.784, 8.203, 'Дело закрыто.'),
    ('408_05_srochno_v_arhiv', 8.932, 10.526, 'Срочно в архив.'),
    ('408_06_sleduyushchee_delo_zavtra', 11.346, 13.564, 'Следующее дело — завтра.'),
    ('d010_A0a_uebb_smotrel_v_pustotu', 14.500, 16.210, 'Уэбб смотрел в пустоту.'),
    ('d010_A0b_pustota_otvetila', 17.008, 18.945, 'Пустота… ответила.'),
    ('d010_A1a_glubokoe_pole_habbla', 19.527, 21.203, 'Глубокое поле Хаббла.'),
    ('d010_A1b_desyat_tysyach_galaktik', 21.608, 23.498, 'Десять тысяч галактик.'),
    ('d010_A2_i_odna_pustaya_tochka', 23.983, 25.603, 'И одна пустая точка.'),
    ('d010_A3a_vsyo_izvestnoe_zakryli', 26.290, 27.728, 'Всё известное закрыли.'),
    ('d010_A3b_ostalos_sem_signalov', 28.116, 29.805, 'Осталось семь сигналов.'),
    ('d010_A4a_shest_galaktiki', 30.145, 31.577, 'Шесть — галактики.'),
    ('d010_A4b_sedmoy_nichey', 31.889, 33.236, 'Седьмой — ничей.'),
    ('d010_A5_odna_liniya_sveta_bolshe_nichego', 33.906, 36.289, 'Одна линия света. Больше — ничего.'),
    ('d010_A6a_avgust_est', 36.996, 38.245, 'Август — есть.'),
    ('d010_A6b_noyabr_drugoy_ugol_est', 38.933, 41.251, 'Ноябрь, другой угол — есть.'),
    ('d010_A7_na_snimkah_habbla_i_uebba_pusto', 42.004, 44.846, 'На снимках Хаббла и Уэбба — пусто.'),
    ('d010_A8_godom_ranshe_ne_bylo_vidno', 45.496, 47.532, 'Годом раньше его там не было видно.'),
    ('d010_A9a_rannyaya_galaktika', 48.155, 49.452, 'Ранняя галактика?'),
    ('d010_A9b_sverhnovaya', 49.933, 50.872, 'Сверхновая?'),
    ('d010_A9c_zvezdu_rvyot_chyornaya_dyra', 51.489, 53.395, 'Звезду рвёт чёрная дыра?'),
    ('d010_A10_smotrim_snova', 54.094, 55.036, 'Смотрим снова.'),
]
EXTRA = [('408_n1_nikto_ne_smog', f'{A}/vo_narrator.mp3', 11.27, 13.03, 'Никто… не смог.')]


def cut(src, a, b, out, prev_end=None, next_start=None):
    a0 = max(0.0, a - 0.05) if prev_end is None else max((prev_end + a) / 2, a - 0.05)
    b0 = b + 0.15 if next_start is None else min((b + next_start) / 2, b + 0.15)
    d = b0 - a0
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{a0:.3f}', '-t', f'{d:.3f}', '-i', src, '-af',
                    f'afade=t=in:d=0.01,afade=t=out:st={d - 0.04:.3f}:d=0.04', '-ar', '48000', '-c:a', 'pcm_s24le', out], check=True)
    return a0, b0


if __name__ == '__main__':
    for i, (n, a, b, tx) in enumerate(SEGS):
        pe = SEGS[i - 1][2] if i else None
        ns = SEGS[i + 1][1] if i + 1 < len(SEGS) else None
        a0, b0 = cut(SRC, a, b, f'{OUT}/{n}.wav', pe, ns)
        print(f'{n}.wav  {a0:.3f}-{b0:.3f}  {tx}')
    for n, src, a, b, tx in EXTRA:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{a}', '-to', f'{b}', '-i', src, '-af',
                        f'afade=t=in:d=0.01,afade=t=out:st={b - a - 0.04:.3f}:d=0.04', '-ar', '48000', '-c:a', 'pcm_s24le',
                        f'{OUT}/{n}.wav'], check=True)
        print(f'{n}.wav  {a}-{b}  {tx}')
