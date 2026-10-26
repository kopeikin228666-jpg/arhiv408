"""Streaming video I/O over ffmpeg pipes (never loads a whole clip in RAM)."""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".exr")


def run(cmd, quiet=True):
    """Run an ffmpeg-ish command, raise with stderr on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode("utf8", "replace")[-3000:])
        raise RuntimeError("command failed: " + " ".join(map(str, cmd))[:400])
    return p


class VideoInfo:
    def __init__(self, w, h, fps, nframes, duration, has_audio=False):
        self.w, self.h, self.fps = int(w), int(h), float(fps)
        self.nframes = int(nframes)
        self.duration = float(duration)
        self.has_audio = has_audio

    def __repr__(self):
        return f"VideoInfo({self.w}x{self.h} @{self.fps:.3f} n={self.nframes} dur={self.duration:.2f}s audio={self.has_audio})"


def _is_sequence(path):
    return any(c in path for c in "*?%") or os.path.isdir(path)


def _seq_files(path):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMG_EXT)]
    elif "%" in path:
        files, i = [], 0
        # printf style: probe from 0 or 1
        for start in (0, 1):
            i = start
            while os.path.exists(path % i):
                files.append(path % i)
                i += 1
            if files:
                break
    else:
        files = glob.glob(path)
    return sorted(files)


def is_image(path):
    return os.path.isfile(path) and path.lower().endswith(IMG_EXT)


def probe(path, fps_default=30.0, still_duration=5.0):
    """Probe a video, an image sequence (dir / glob / printf pattern) or a single still."""
    if _is_sequence(path):
        files = _seq_files(path)
        if not files:
            raise FileNotFoundError(path)
        im = cv2.imread(files[0], cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        n = len(files)
        return VideoInfo(w, h, fps_default, n, n / fps_default)
    if is_image(path):
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        n = int(round(still_duration * fps_default))
        return VideoInfo(w, h, fps_default, n, n / fps_default)
    out = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if out.returncode != 0:
        raise RuntimeError("ffprobe failed on %s: %s" % (path, out.stderr.decode()[-500:]))
    js = json.loads(out.stdout)
    vs = [s for s in js["streams"] if s["codec_type"] == "video"]
    if not vs:
        raise RuntimeError("no video stream in " + path)
    v = vs[0]
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else fps_default
    if fps <= 0 or fps > 240:
        num, den = v.get("r_frame_rate", "30/1").split("/")
        fps = float(num) / float(den)
    dur = float(v.get("duration") or js["format"].get("duration") or 0)
    n = int(v.get("nb_frames") or 0) or int(round(dur * fps))
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) in (90, 270):
        w, h = h, w
    has_audio = any(s["codec_type"] == "audio" for s in js["streams"])
    return VideoInfo(w, h, fps, n, dur or n / fps, has_audio)


class FrameReader:
    """Iterate BGR uint8 frames from a video / image sequence / still image.

    size=(w,h) rescales on decode (ffmpeg does it, cheap). fps forces a frame rate.
    """

    def __init__(self, path, size=None, fps=None, still_duration=5.0, loop=False):
        self.path = path
        self.info = probe(path, fps_default=fps or 30.0, still_duration=still_duration)
        self.size = size or (self.info.w, self.info.h)
        self.fps = fps or self.info.fps
        self.loop = loop
        self._proc = None
        self._files = None
        self._still = None
        if _is_sequence(path):
            self._files = _seq_files(path)
        elif is_image(path):
            self._still = self._load_img(path)

    def _load_img(self, p):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if (im.shape[1], im.shape[0]) != tuple(self.size):
            im = cv2.resize(im, tuple(self.size), interpolation=cv2.INTER_AREA)
        return im

    def __iter__(self):
        if self._still is not None:
            while True:
                for _ in range(self.info.nframes):
                    yield self._still.copy()
                if not self.loop:
                    return
        if self._files is not None:
            while True:
                for f in self._files:
                    yield self._load_img(f)
                if not self.loop:
                    return
        while True:
            yield from self._iter_video()
            if not self.loop:
                return

    def _iter_video(self):
        w, h = self.size
        vf = []
        if (w, h) != (self.info.w, self.info.h):
            vf.append(f"scale={w}:{h}:flags=lanczos")
        cmd = [FFMPEG, "-v", "error", "-nostdin", "-i", self.path]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        if self.fps and abs(self.fps - self.info.fps) > 1e-3:
            cmd += ["-r", str(self.fps)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
        self._proc = proc
        nbytes = w * h * 3
        try:
            while True:
                buf = proc.stdout.read(nbytes)
                if len(buf) < nbytes:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        finally:
            proc.stdout.close()
            proc.wait()


class FrameWriter:
    """Pipe BGR uint8 frames into ffmpeg -> H.264 yuv420p (crf 14 by default).

    audio_from: optional path whose audio track is muxed in (if it has one).
    lossless=True writes a fast lossless intermediate (x264 qp0, yuv444p) for chains.
    """

    def __init__(self, path, w, h, fps, crf=14, preset="medium", tune=None, audio_from=None,
                 lossless=False, extra=None):
        self.path, self.w, self.h = path, w, h
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        cmd = [FFMPEG, "-v", "error", "-y", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
        use_audio = False
        if audio_from and not lossless:
            try:
                if not _is_sequence(audio_from) and not is_image(audio_from) and probe(audio_from).has_audio:
                    use_audio = True
            except Exception:
                use_audio = False
        if use_audio:
            cmd += ["-i", audio_from, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "256k", "-shortest"]
        if lossless:
            cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p"]
        else:
            cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-movflags", "+faststart"]
            if tune:
                cmd += ["-tune", tune]
        if extra:
            cmd += list(extra)
        cmd += [path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=w * h * 3 * 2)
        self.count = 0

    def write(self, frame):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.count += 1

    def close(self):
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError("ffmpeg encoder failed for " + self.path)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def encode_args(crf=14, preset="medium"):
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-movflags", "+faststart"]


def to_video_file(path, tmpdir, fps=30.0, size=None):
    """Make sure `path` is a real video file (sequences / stills are encoded losslessly)."""
    if not _is_sequence(path) and not is_image(path):
        return path
    info = probe(path, fps_default=fps)
    out = os.path.join(tmpdir, "seq_%d.mkv" % (abs(hash(path)) % 10**8))
    r = FrameReader(path, size=size, fps=fps)
    w, h = r.size
    wr = FrameWriter(out, w, h, fps, lossless=True)
    for f in r:
        wr.write(f)
    wr.close()
    return out
