#!/usr/bin/env python3
"""
pushup_scapula_tracker.py
─────────────────────────────────────────────────────────────────────────────
Live scapulothoracic joint tracker during push-ups using a webcam.

MEASURED  : Scapular upward rotation angle  (both sides, degrees)
COMPUTED  : Scapulohumeral Rhythm ratio      (ΔGH / Δscap)
MODEL     : Rule-based correctness classifier

Works with a regular laptop webcam or any USB/IP camera.
No model training needed — uses Google's pre-trained MediaPipe Pose.

Literature basis:
  Ludewig & Cook 2000  JOSPT  — scapular motion norms during push-up
  McClure et al. 2001  JSES   — 30-45° upward rotation at top of push-up
  Ludewig et al. 2004  AJSM   — serratus anterior balance
  Inman 1944 / Bagg & Forrest 1988 — SHR ~2:1 in healthy subjects

Setup:
  pip install opencv-python mediapipe numpy

Run:
  python pushup_scapula_tracker.py

Controls:
  Q / ESC   quit
  R         reset rep counter
  S         save screenshot
  C         toggle HSV colour-marker overlay
  H         toggle help panel
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import math
import os
import csv
import datetime
from collections import deque


# ── dependency check ─────────────────────────────────────────────────────────

_missing = []
for _pkg, _mod in [("opencv-python", "cv2"),
                   ("mediapipe",     "mediapipe"),
                   ("numpy",         "numpy")]:
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    sys.exit(
        f"\nMissing packages. Run:\n  pip install {' '.join(_missing)}\n"
    )

import cv2
import mediapipe as mp
import numpy as np


# ── MediaPipe ────────────────────────────────────────────────────────────────

_pose_sol    = mp.solutions.pose
_draw_utils  = mp.solutions.drawing_utils
_draw_styles = mp.solutions.drawing_styles

# landmark indices we care about
SH_L,  SH_R  = 11, 12
EL_L,  EL_R  = 13, 14
WR_L,  WR_R  = 15, 16
HIP_L, HIP_R = 23, 24


# ── clinical thresholds (directly from literature) ───────────────────────────

SCAP_MIN  = 20.0   # Ludewig & Cook 2000 — lower bound, upward rotation
SCAP_MAX  = 50.0   # upper bound
SHR_LO    = 1.4    # Inman 1944 / Bagg & Forrest 1988
SHR_HI    = 2.8
WING_THR  = 12.0   # left-right asymmetry → flag possible winging
EL_BOTTOM = 105    # elbow angle at bottom of push-up
EL_TOP    = 155    # elbow angle at top (baseline capture point)


# ── BGR colour constants ─────────────────────────────────────────────────────

WHT = (255, 255, 255)
BLK = (  0,   0,   0)
GRN = ( 50, 220,  80)
RED = ( 60,  60, 220)
YLW = ( 30, 220, 220)
ORG = ( 30, 160, 255)
BLU = (240, 130,  30)
CYN = (230, 220,  30)
GRY = (160, 160, 160)
PNK = (180, 100, 220)


# ── HSV ranges for adhesive dot detection ────────────────────────────────────
# Optional hardware add-on: stick coloured dots on bony landmarks.
# Green  → inferior scapular angle
# Red    → acromion
# Blue   → T-spine reference (T3 or T7)
# Yellow → lateral epicondyle

HSV_DOTS = {
    "inf_angle": {
        "lo":  np.array([ 35, 120, 120]),
        "hi":  np.array([ 85, 255, 255]),
        "col": GRN, "label": "Inf Ang",
    },
    "acromion": {
        "lo":  np.array([  0, 150, 150]),
        "hi":  np.array([ 15, 255, 255]),
        "col": RED, "label": "Acromion",
    },
    "t_spine": {
        "lo":  np.array([100, 120, 120]),
        "hi":  np.array([140, 255, 255]),
        "col": BLU, "label": "T-Spine",
    },
    "lat_epi": {
        "lo":  np.array([ 20, 150, 150]),
        "hi":  np.array([ 35, 255, 255]),
        "col": YLW, "label": "Lat Epi",
    },
}


# ── geometry helpers ──────────────────────────────────────────────────────────

def angle3(a, b, c):
    """Angle at vertex b, in degrees. Works on any 2-D or 3-D point lists."""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def midpt(a, b):
    return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]


def running_mean(buf: deque, val: float) -> float:
    buf.append(val)
    return float(np.mean(buf))


def scap_rot_proxy(sh, hip, opp_sh):
    """
    2-D proxy for scapular upward rotation — posterior camera view.

    MediaPipe doesn't expose the inferior scapular angle or true acromion
    separately, so we use the shoulder landmark (which sits close to the
    acromion in this view) and measure its angular deviation from the
    thoracic axis (hip → mid-shoulder midpoint).

    This matches the approach in 2-D video-based kinematics work and is
    consistent with Ludewig & Cook (2000) who report ~30-45° at the top
    of a push-up using a similar line-of-sight proxy.
    """
    mid_sh = midpt(sh, opp_sh)
    thorax = np.array([mid_sh[0] - hip[0], mid_sh[1] - hip[1]], float)
    sh_vec = np.array([sh[0]    - hip[0], sh[1]    - hip[1]],   float)
    denom  = np.linalg.norm(thorax) * np.linalg.norm(sh_vec)
    if denom < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(thorax, sh_vec) / denom, -1.0, 1.0))))


# ── HSV dot detection ─────────────────────────────────────────────────────────

def detect_dots(frame_bgr):
    """Return {name: (cx, cy)} for every colour sticker found in frame."""
    hsv  = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), (9, 9), 0)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    out  = {}
    for name, cfg in HSV_DOTS.items():
        mask = cv2.inRange(hsv, cfg["lo"], cfg["hi"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        biggest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(biggest) < 150:
            continue
        M = cv2.moments(biggest)
        if M["m00"] > 0:
            out[name] = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return out


def draw_dots(frame, dots):
    for name, (cx, cy) in dots.items():
        cfg = HSV_DOTS[name]
        cv2.circle(frame, (cx, cy), 12, cfg["col"], -1)
        cv2.circle(frame, (cx, cy), 14, WHT, 2)
        cv2.putText(frame, cfg["label"], (cx + 16, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cfg["col"], 1)
    # if both scapular dots are visible, draw the actual scapular axis line
    if "inf_angle" in dots and "acromion" in dots:
        ia = np.array(dots["inf_angle"], float)
        ac = np.array(dots["acromion"],  float)
        cv2.line(frame, dots["inf_angle"], dots["acromion"], CYN, 2)
        ang = math.degrees(math.atan2(abs(ac[0] - ia[0]), abs(ac[1] - ia[1])))
        mid = tuple(((ia + ac) / 2).astype(int))
        cv2.putText(frame, f"axis {ang:.1f}d", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, CYN, 1)


# ── correctness classifier ────────────────────────────────────────────────────

def classify(sl, sr, shr_l, shr_r, el_l, el_r):
    """
    Rule-based model using thresholds from literature.
    Returns (status_string, [issue_strings], is_correct).
    """
    issues = []
    ok     = True
    avg_s  = (sl + sr) / 2.0

    # scapular rotation range
    if avg_s < SCAP_MIN:
        issues.append(f"Insufficient scap rotation  ({avg_s:.0f} < {SCAP_MIN}\u00b0)")
        ok = False
    elif avg_s > SCAP_MAX:
        issues.append(f"Excessive scap elevation    ({avg_s:.0f} > {SCAP_MAX}\u00b0)")
        ok = False

    # left-right winging check
    asym = abs(sl - sr)
    if asym > WING_THR:
        side = "LEFT" if sl < sr else "RIGHT"
        issues.append(f"Possible winging — {side}  ({asym:.0f}\u00b0 asymmetry)")
        ok = False

    # scapulohumeral rhythm
    for label, shr in [("L", shr_l), ("R", shr_r)]:
        if shr is not None:
            if shr < SHR_LO:
                issues.append(f"SHR-{label} {shr:.2f}: scapula lagging humerus")
                ok = False
            elif shr > SHR_HI:
                issues.append(f"SHR-{label} {shr:.2f}: scapula over-rotating")
                ok = False

    # elbow depth (informational only)
    avg_el = (el_l + el_r) / 2.0
    if avg_el > 165:
        issues.append("Arms nearly straight — outside push-up range")
    elif avg_el < 55:
        issues.append("Excessive depth — impingement risk")

    return ("CORRECT \u2713" if ok else "INCORRECT \u2717"), issues, ok


# ── rep quality score (0–100) ─────────────────────────────────────────────────

def quality_score(sl, sr, shr_l, shr_r):
    """
    Simple linear score centred on the middle of each clinical range.
    100 = dead-centre on every parameter. Not published — just a useful
    single-number summary for the user.
    """
    pts = []
    avg_s    = (sl + sr) / 2.0
    mid_s    = (SCAP_MIN + SCAP_MAX) / 2.0
    half_s   = (SCAP_MAX - SCAP_MIN) / 2.0
    pts.append(max(0.0, 100.0 - abs(avg_s - mid_s) / half_s * 50.0))
    pts.append(max(0.0, 100.0 - abs(sl - sr) / WING_THR * 50.0))
    mid_shr  = (SHR_LO + SHR_HI) / 2.0
    half_shr = (SHR_HI - SHR_LO) / 2.0
    for shr in [shr_l, shr_r]:
        if shr is not None:
            pts.append(max(0.0, 100.0 - abs(shr - mid_shr) / half_shr * 50.0))
    return round(sum(pts) / len(pts)) if pts else 50


# ── sparkline mini-graph ──────────────────────────────────────────────────────

class Sparkline:
    """Rolling line graph drawn directly onto a frame with cv2."""

    def __init__(self, maxlen=120, lo=0.0, hi=60.0, width=200, height=50):
        self.buf   = deque(maxlen=maxlen)
        self.lo    = lo
        self.hi    = hi
        self.w     = width
        self.h     = height

    def push(self, v):
        self.buf.append(float(v))

    def draw(self, frame, x, y, label, col):
        if len(self.buf) < 2:
            return
        pts = []
        n = len(self.buf)
        for i, v in enumerate(self.buf):
            px_ = x + int(i / (n - 1) * (self.w - 1))
            frac = float(np.clip((v - self.lo) / (self.hi - self.lo), 0, 1))
            py_ = y + self.h - int(frac * (self.h - 1))
            pts.append((px_, py_))

        # background box
        cv2.rectangle(frame, (x, y), (x + self.w, y + self.h), (20, 20, 20), -1)
        cv2.rectangle(frame, (x, y), (x + self.w, y + self.h), GRY, 1)

        # reference lines at clinical bounds
        for v_ref in [SCAP_MIN, SCAP_MAX]:
            frac = float(np.clip((v_ref - self.lo) / (self.hi - self.lo), 0, 1))
            py_  = y + self.h - int(frac * (self.h - 1))
            cv2.line(frame, (x, py_), (x + self.w, py_), (80, 80, 80), 1)

        # the actual line
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], col, 1)

        cv2.putText(frame, label, (x + 3, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)


# ── main tracker class ────────────────────────────────────────────────────────

class PushupTracker:

    def __init__(self):
        # model_complexity=1 gives a good speed/accuracy tradeoff on a laptop
        self.pose = _pose_sol.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        # smoothing buffers — 6-frame moving average kills webcam jitter
        self.bufs = {k: deque(maxlen=6) for k in
                     ["sl", "sr", "el_l", "el_r", "gh_l", "gh_r"]}

        # SHR baseline — captured once per rep at the top position
        self.base_sl   = self.base_sr   = None
        self.base_gh_l = self.base_gh_r = None
        self.shr_l     = self.shr_r     = None

        # rep state machine
        self.reps  = 0
        self.phase = "UP"
        self._lock = False
        self.log   = []

        # sparklines
        self.spark_l = Sparkline(maxlen=120, lo=0.0, hi=60.0)
        self.spark_r = Sparkline(maxlen=120, lo=0.0, hi=60.0)

        # CSV session log
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"session_{ts}.csv"
        self._csv_f   = open(self.csv_path, "w", newline="")
        self._csv_w   = csv.writer(self._csv_f)
        self._csv_w.writerow(["rep", "scap_l", "scap_r", "shr_l", "shr_r",
                               "el_l", "el_r", "asym", "score", "correct"])

        # fps tracking
        self.fps   = 0.0
        self._ft   = time.time()
        self._fc   = 0
        self.ss_n  = 0

    # ── landmark helpers ──────────────────────────────────────────────────────

    def _lm(self, lms, idx):
        lm = lms.landmark[idx]
        return [lm.x, lm.y]

    def _vis(self, lms, idx):
        return getattr(lms.landmark[idx], "visibility", 1.0)

    # ── rep state machine ─────────────────────────────────────────────────────

    def _tick_rep(self, el_avg, sl, sr, el_l, el_r, ok):
        if self.phase == "UP" and el_avg < EL_BOTTOM:
            self.phase = "DOWN"
            self._lock = False

        elif self.phase == "DOWN" and el_avg > EL_TOP and not self._lock:
            self.phase = "UP"
            self._lock = True
            self.reps += 1
            score = quality_score(sl, sr, self.shr_l, self.shr_r)
            rec = {
                "rep":    self.reps,
                "scap_l": round(sl,  1),
                "scap_r": round(sr,  1),
                "shr_l":  round(self.shr_l, 2) if self.shr_l is not None else None,
                "shr_r":  round(self.shr_r, 2) if self.shr_r is not None else None,
                "el_l":   round(el_l, 1),
                "el_r":   round(el_r, 1),
                "asym":   round(abs(sl - sr), 1),
                "score":  score,
                "correct": ok,
            }
            self.log.append(rec)
            self._csv_w.writerow([rec[k] for k in
                ["rep", "scap_l", "scap_r", "shr_l", "shr_r",
                 "el_l", "el_r", "asym", "score", "correct"]])
            self._csv_f.flush()
            self._reset_baseline()

    def _reset_baseline(self):
        self.base_sl   = self.base_sr   = None
        self.base_gh_l = self.base_gh_r = None
        self.shr_l     = self.shr_r     = None

    def reset(self):
        self.reps  = 0
        self.phase = "UP"
        self._lock = False
        self.log   = []
        self._reset_baseline()

    # ── per-frame processing ──────────────────────────────────────────────────

    def process(self, frame, show_dots=False, show_help=False):
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.pose.process(rgb)
        rgb.flags.writeable = True

        sl = sr = 0.0
        el_l = el_r = 180.0
        status  = "DETECTING..."
        issues  = []
        ok      = False

        if show_dots:
            dots = detect_dots(frame)
            draw_dots(frame, dots)

        if res.pose_landmarks:
            lms    = res.pose_landmarks
            vis_ok = all(self._vis(lms, i) > 0.35
                         for i in [SH_L, SH_R, EL_L, EL_R, HIP_L, HIP_R])

            if vis_ok:
                l_sh  = self._lm(lms, SH_L);  r_sh  = self._lm(lms, SH_R)
                l_el  = self._lm(lms, EL_L);  r_el  = self._lm(lms, EL_R)
                l_wr  = self._lm(lms, WR_L);  r_wr  = self._lm(lms, WR_R)
                l_hip = self._lm(lms, HIP_L); r_hip = self._lm(lms, HIP_R)
                mhip  = midpt(l_hip, r_hip)

                # MEASURED — scapular upward rotation (proxy, posterior view)
                sl = running_mean(self.bufs["sl"],
                                  scap_rot_proxy(l_sh, l_hip, r_sh))
                sr = running_mean(self.bufs["sr"],
                                  scap_rot_proxy(r_sh, r_hip, l_sh))

                # elbow angles
                el_l = running_mean(self.bufs["el_l"], angle3(l_sh, l_el, l_wr))
                el_r = running_mean(self.bufs["el_r"], angle3(r_sh, r_el, r_wr))

                # glenohumeral angles (mid-hip → shoulder → elbow)
                gh_l = running_mean(self.bufs["gh_l"], angle3(mhip, l_sh, l_el))
                gh_r = running_mean(self.bufs["gh_r"], angle3(mhip, r_sh, r_el))

                # COMPUTED — SHR baseline captured when arms are extended
                if el_l > EL_TOP and self.base_sl is None:
                    self.base_sl   = sl;   self.base_sr   = sr
                    self.base_gh_l = gh_l; self.base_gh_r = gh_r

                # COMPUTED — SHR ratio
                if self.base_sl is not None:
                    d_sl = sl - self.base_sl
                    d_sr = sr - self.base_sr
                    # only compute when there's enough delta to avoid div-by-noise
                    if abs(d_sl) > 1.0:
                        self.shr_l = (gh_l - self.base_gh_l) / d_sl
                    if abs(d_sr) > 1.0:
                        self.shr_r = (gh_r - self.base_gh_r) / d_sr

                status, issues, ok = classify(sl, sr, self.shr_l, self.shr_r,
                                              el_l, el_r)
                self._tick_rep((el_l + el_r) / 2.0, sl, sr, el_l, el_r, ok)

                self.spark_l.push(sl)
                self.spark_r.push(sr)

                self._draw_skeleton(frame, lms, w, h,
                                    l_sh, r_sh, l_el, r_el,
                                    l_wr, r_wr, l_hip, r_hip, sl, sr)
            else:
                status = "LOW CONFIDENCE — reposition or improve lighting"
        else:
            status = "NO POSE DETECTED"
            issues = ["Move closer to camera or improve lighting"]

        self._draw_hud(frame, sl, sr, el_l, el_r, status, issues, ok, w, h)

        if show_help:
            self._draw_help(frame, w, h)

        # fps counter
        self._fc += 1
        t = time.time()
        if t - self._ft >= 1.0:
            self.fps = self._fc / (t - self._ft)
            self._fc = 0
            self._ft = t
        cv2.putText(frame, f"FPS {self.fps:.0f}",
                    (w - 85, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYN, 1)

        return frame

    # ── skeleton overlay ──────────────────────────────────────────────────────

    def _draw_skeleton(self, frame, lms, w, h,
                       l_sh, r_sh, l_el, r_el,
                       l_wr, r_wr, l_hip, r_hip, sl, sr):

        def px(lm_norm):
            return (int(lm_norm[0] * w), int(lm_norm[1] * h))

        p = {
            "lsh": px(l_sh), "rsh": px(r_sh),
            "lel": px(l_el), "rel": px(r_el),
            "lwr": px(l_wr), "rwr": px(r_wr),
            "lhp": px(l_hip),"rhp": px(r_hip),
        }

        segs = [
            ("lsh", "rsh",  YLW, 2),
            ("lhp", "rhp",  YLW, 2),
            ("lsh", "lhp",  GRY, 1),
            ("rsh", "rhp",  GRY, 1),
            ("lsh", "lel",  ORG, 3),
            ("lel", "lwr",  ORG, 3),
            ("rsh", "rel",  ORG, 3),
            ("rel", "rwr",  ORG, 3),
        ]
        for a, b, col, thick in segs:
            cv2.line(frame, p[a], p[b], col, thick)

        # shoulder landmarks — colour shows whether scap rotation is in range
        for side, pt, sv in [("L", p["lsh"], sl), ("R", p["rsh"], sr)]:
            dot_col = GRN if SCAP_MIN <= sv <= SCAP_MAX else RED
            cv2.circle(frame, pt, 10, dot_col, -1)
            cv2.circle(frame, pt, 12, WHT, 1)
            cv2.putText(frame, f"Acr {side}", (pt[0] + 14, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot_col, 1)

        for pt in (p["lel"], p["rel"]):
            cv2.circle(frame, pt, 8, GRN, -1)
            cv2.circle(frame, pt, 9, WHT, 1)

        for pt in (p["lhp"], p["rhp"]):
            cv2.circle(frame, pt, 7, PNK, -1)

        # scapular axis proxy line + angle label
        for sh_pt, hp_pt, sv in [(p["lsh"], p["lhp"], sl),
                                  (p["rsh"], p["rhp"], sr)]:
            cv2.line(frame, sh_pt, hp_pt, (80, 80, 180), 1)
            mid = ((sh_pt[0] + hp_pt[0]) // 2, (sh_pt[1] + hp_pt[1]) // 2)
            cv2.putText(frame, f"{sv:.0f}\u00b0", (mid[0] + 6, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYN, 1)

    # ── HUD ──────────────────────────────────────────────────────────────────

    def _draw_hud(self, frame, sl, sr, el_l, el_r,
                  status, issues, ok, w, h):

        # semi-transparent left panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (382, 340), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, "SCAPULA PUSH-UP TRACKER",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHT, 2)
        cv2.line(frame, (8, 40), (382, 40), GRY, 1)

        def row(label, vl, vr, y, ok_l=True, ok_r=True):
            cv2.putText(frame, label + ":", (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, GRY, 1)
            cv2.putText(frame, f"L {vl}", (186, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, GRN if ok_l else RED, 1)
            cv2.putText(frame, f"R {vr}", (288, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, GRN if ok_r else RED, 1)

        def inr(v, lo, hi):
            return lo <= v <= hi

        row("Scap Rot (\u00b0)",
            f"{sl:.1f}", f"{sr:.1f}", 68,
            inr(sl, SCAP_MIN, SCAP_MAX),
            inr(sr, SCAP_MIN, SCAP_MAX))

        def fmt_shr(v):
            return f"{v:.2f}" if v is not None else "--"
        def shr_in_range(v):
            return v is not None and inr(v, SHR_LO, SHR_HI)

        row("SHR ratio",
            fmt_shr(self.shr_l), fmt_shr(self.shr_r), 94,
            shr_in_range(self.shr_l), shr_in_range(self.shr_r))

        row("Elbow (\u00b0)",
            f"{el_l:.1f}", f"{el_r:.1f}", 120, True, True)

        asym = abs(sl - sr)
        cv2.putText(frame,
                    f"Asymmetry: {asym:.1f}\u00b0   Phase: {self.phase}",
                    (18, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    GRN if asym <= WING_THR else RED, 1)

        cv2.line(frame, (8, 156), (382, 156), GRY, 1)

        cv2.putText(frame, f"Reps: {self.reps}",
                    (18, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.56, YLW, 1)

        bset = self.base_sl is not None
        cv2.putText(frame,
                    f"SHR base: {'SET' if bset else 'WAITING \u2014 extend arms fully'}",
                    (18, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    GRN if bset else GRY, 1)

        if self.log:
            last  = self.log[-1]
            sc    = last["score"]
            sc_c  = GRN if sc >= 75 else (YLW if sc >= 50 else RED)
            cv2.putText(frame, f"Last rep score: {sc}/100",
                        (18, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.44, sc_c, 1)

        cv2.line(frame, (8, 228), (382, 228), GRY, 1)

        # status bar — green=correct, red=incorrect, yellow=detecting
        s_col = GRN if ok else (RED if "INCORRECT" in status else YLW)
        cv2.rectangle(frame, (8, 235), (382, 265), s_col, -1)
        cv2.putText(frame, status, (18, 257),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, BLK, 2)

        if issues:
            iy  = 273
            n   = min(len(issues), 3)
            ov2 = frame.copy()
            cv2.rectangle(ov2, (8, iy - 4), (382, iy + n * 24 + 6),
                          (10, 10, 60), -1)
            cv2.addWeighted(ov2, 0.65, frame, 0.35, 0, frame)
            for i, msg in enumerate(issues[:3]):
                cv2.putText(frame, f"  {msg}", (12, iy + 18 + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1)

        # sparklines — bottom-left corner
        sy = h - 115
        self.spark_l.draw(frame, 10,  sy, "Scap-L", GRN)
        self.spark_r.draw(frame, 220, sy, "Scap-R", ORG)

        # reference bar and controls — very bottom
        cv2.putText(frame,
                    f"Scap {SCAP_MIN}\u2013{SCAP_MAX}\u00b0  "
                    f"SHR {SHR_LO}\u2013{SHR_HI}  "
                    f"Wing <{WING_THR}\u00b0",
                    (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.37, GRY, 1)
        cv2.putText(frame,
                    "Q=Quit  R=Reset  S=Screenshot  C=Markers  H=Help",
                    (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.37, GRY, 1)

    # ── help panel ────────────────────────────────────────────────────────────

    def _draw_help(self, frame, w, h):
        lines = [
            "\u2500\u2500 CAMERA SETUP \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            "Posterior view (camera faces subject's BACK)",
            "Shoulder height, 1.5\u20132 m from camera",
            "Full upper body + hips must be in frame",
            "\u2500\u2500 OPTIONAL COLOUR STICKERS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            "Green  = Inferior scapular angle",
            "Red    = Acromion",
            "Blue   = T3/T7 spinous process",
            "Yellow = Lateral epicondyle",
            "\u2500\u2500 CLINICAL RANGES \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            f"Scap rotation : {SCAP_MIN}\u2013{SCAP_MAX}\u00b0  (Ludewig & Cook 2000)",
            f"SHR ratio     : {SHR_LO}\u2013{SHR_HI}  normal \u22482:1",
            f"Winging flag  : >{WING_THR}\u00b0 L\u2013R asymmetry",
            "\u2500\u2500 LITERATURE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            "Ludewig & Cook (2000) JOSPT",
            "McClure et al.  (2001) JSES",
            "Ludewig et al.  (2004) AJSM",
            "Inman (1944) / Bagg & Forrest (1988)",
            "\u2500\u2500 CONTROLS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            "Q/ESC quit    R reset    S screenshot",
            "C colour-markers toggle   H this panel",
        ]
        px0 = w - 380
        py0 = 10
        ph  = len(lines) * 18 + 14
        ov  = frame.copy()
        cv2.rectangle(ov, (px0 - 5, py0 - 5), (px0 + 368, py0 + ph),
                      (15, 15, 15), -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
        for i, line in enumerate(lines):
            col = YLW if line.startswith("\u2500") else GRY
            cv2.putText(frame, line, (px0, py0 + 13 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

    # ── session summary ───────────────────────────────────────────────────────

    def summary(self):
        print("\n" + "=" * 62)
        print("  SESSION SUMMARY")
        print("=" * 62)
        print(f"  Total reps : {self.reps}")
        if self.log:
            good      = sum(1 for r in self.log if r["correct"])
            avg_score = sum(r["score"] for r in self.log) / len(self.log)
            print(f"  Correct    : {good} / {len(self.log)}")
            print(f"  Avg score  : {avg_score:.0f} / 100")
            print()
            print(f"  {'Rep':>4}  {'ScapL':>6}  {'ScapR':>6}  "
                  f"{'SHR-L':>6}  {'SHR-R':>6}  {'Score':>5}  {'OK?':>5}")
            print("  " + "-" * 52)
            for r in self.log:
                sl_s = f"{r['shr_l']:.2f}" if r["shr_l"] is not None else "  --"
                sr_s = f"{r['shr_r']:.2f}" if r["shr_r"] is not None else "  --"
                print(f"  {r['rep']:>4}  {r['scap_l']:>6.1f}  {r['scap_r']:>6.1f}  "
                      f"{sl_s:>6}  {sr_s:>6}  {r['score']:>5}  "
                      f"{'YES' if r['correct'] else 'NO':>5}")
        print()
        print(f"  Log saved → {self.csv_path}")
        print("=" * 62 + "\n")

    def close(self):
        self.pose.close()
        self._csv_f.close()


# ── camera helpers ────────────────────────────────────────────────────────────

def pick_camera():
    print()
    print("=" * 56)
    print("  SCAPULA PUSH-UP TRACKER  —  Camera")
    print("=" * 56)
    print("  0    Built-in / laptop webcam     (default)")
    print("  1    First external USB camera")
    print("  2    Second external USB camera")
    print("  URL  e.g.  http://192.168.1.x:8080/video")
    print()
    try:
        c = input("  Choice [0]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0
    if c in ("", "0"):
        return 0
    if c.isdigit():
        return int(c)
    return c


def open_cam(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    src = pick_camera()
    print(f"\n  Opening camera {src!r} …")
    cap = open_cam(src)
    if cap is None:
        sys.exit(f"  ERROR: Cannot open camera '{src}'. Try 0, 1, or 2.")

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera ready: {aw} \u00d7 {ah}")
    print()
    print("  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
    print("  \u2502  Camera should face the subject's BACK         \u2502")
    print("  \u2502  Full upper body + hips must be visible        \u2502")
    print("  \u2502  Recommended: 1.5\u20132 m away, shoulder height     \u2502")
    print("  \u2502  Press H in the window for on-screen help      \u2502")
    print("  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")
    print()

    tracker   = PushupTracker()
    show_dots = False
    show_help = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  Frame read failed — check camera connection.")
                break

            frame = tracker.process(frame,
                                    show_dots=show_dots,
                                    show_help=show_help)
            cv2.imshow("Scapula Push-up Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):       # quit
                break
            elif key in (ord("r"), ord("R")):          # reset
                tracker.reset()
                print("  [R] Rep counter + SHR baseline cleared.")
            elif key in (ord("s"), ord("S")):          # screenshot
                tracker.ss_n += 1
                fn = f"scapula_{tracker.ss_n:03d}.png"
                cv2.imwrite(fn, frame)
                print(f"  [S] Screenshot saved \u2192 {fn}")
            elif key in (ord("c"), ord("C")):          # colour markers
                show_dots = not show_dots
                print(f"  [C] Colour markers {'ON' if show_dots else 'OFF'}")
            elif key in (ord("h"), ord("H")):          # help
                show_help = not show_help

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        tracker.summary()


if __name__ == "__main__":
    main()
