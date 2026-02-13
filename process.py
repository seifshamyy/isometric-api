"""
process.py  –  Core isometric rendering logic.
Exposes a single function: generate_isometric(img_bgr) -> PNG bytes
"""

import io
import re
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pytesseract

# ── CONFIG ────────────────────────────────────────────────────────────────────
WALL_HEIGHT    = 0.5
APPROX_EPSILON = 0.02
MAX_DIM        = 4.0
CAM_ELEV       = 48
CAM_AZIM       = -30

BG_COLOR    = "#dde0e5"
FLOOR_COLOR = "#c8c8c8"
FLOOR_EDGE  = "#c8c8c8"
TONE_RIGHT  = "#909090"
TONE_LEFT   = "#7a7a7a"
EDGE_RIGHT  = "#707070"
EDGE_LEFT   = "#2a2a2a"


def generate_isometric(img_bgr: np.ndarray) -> bytes:
    """
    Takes an OpenCV BGR image array, returns PNG bytes of the isometric render.
    """
    h_img, w_img = img_bgr.shape[:2]

    # ── 1. PRE-PROCESS ────────────────────────────────────────────────────────
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize=15, C=4)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # ── 2. FIND LARGEST POLYGON ───────────────────────────────────────────────
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image.")

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    best_points = None
    for cnt in contours_sorted[:5]:
        if cv2.contourArea(cnt) < h_img * w_img * 0.02:
            continue
        eps    = APPROX_EPSILON * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if 3 <= len(approx) <= 12:
            best_points = approx.reshape(-1, 2).astype(float)
            break

    if best_points is None:
        cnt         = contours_sorted[0]
        hull        = cv2.convexHull(cnt)
        eps         = APPROX_EPSILON * cv2.arcLength(hull, True)
        approx      = cv2.approxPolyDP(hull, eps, True)
        best_points = approx.reshape(-1, 2).astype(float)

    # ── 3. NORMALISE → metres ─────────────────────────────────────────────────
    pts = best_points.copy()
    pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min() + 1e-9)
    pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min() + 1e-9)
    scale = MAX_DIM / max(pts[:, 0].max() - pts[:, 0].min(),
                          pts[:, 1].max() - pts[:, 1].min())
    pts_m = pts * scale

    span_x = pts_m[:, 0].max() - pts_m[:, 0].min()
    span_y = pts_m[:, 1].max() - pts_m[:, 1].min()
    wall_h = WALL_HEIGHT * (span_x + span_y) / 2.0

    # ── 4. BUILD GEOMETRY ─────────────────────────────────────────────────────
    n        = len(pts_m)
    centroid = pts_m.mean(axis=0)

    floor_pts   = np.column_stack([pts_m[:, 0], pts_m[:, 1], np.zeros(n)])
    ceiling_pts = np.column_stack([pts_m[:, 0], pts_m[:, 1], np.full(n, wall_h)])

    wall_faces, wall_normals = [], []
    for i in range(n):
        j = (i + 1) % n
        wall_faces.append([floor_pts[i], floor_pts[j],
                           ceiling_pts[j], ceiling_pts[i]])
        dx = pts_m[j, 0] - pts_m[i, 0]
        dy = pts_m[j, 1] - pts_m[i, 1]
        nx, ny = dy, -dx
        mid = (pts_m[i] + pts_m[j]) / 2
        if (nx * (mid[0] - centroid[0]) + ny * (mid[1] - centroid[1])) < 0:
            nx, ny = -nx, -ny
        length = np.hypot(nx, ny) + 1e-9
        wall_normals.append(np.array([nx / length, ny / length, 0.0]))

    elev_r  = np.radians(CAM_ELEV)
    azim_r  = np.radians(CAM_AZIM)
    cam_dir   = np.array([np.cos(elev_r) * np.cos(azim_r),
                          np.cos(elev_r) * np.sin(azim_r),
                          np.sin(elev_r)])
    cam_right = np.array([np.sin(azim_r), -np.cos(azim_r), 0.0])

    def faces_camera(normal):
        return float(np.dot(normal, cam_dir)) > 0.0

    def wall_shade(normal, is_front):
        is_right = float(np.dot(normal[:2], cam_right[:2])) >= 0
        fc = TONE_RIGHT if is_right else TONE_LEFT
        ec = EDGE_RIGHT if is_right else EDGE_LEFT
        return fc, ec, (0.28 if is_front else 1.0)

    back_walls  = [(i, q) for i, q in enumerate(wall_faces) if not faces_camera(wall_normals[i])]
    front_walls = [(i, q) for i, q in enumerate(wall_faces) if     faces_camera(wall_normals[i])]

    # ── 5. RENDER ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9), facecolor=BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_COLOR)
    ax.grid(False)
    ax.set_axis_off()
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("none")

    # Floor
    ax.add_collection3d(Poly3DCollection([floor_pts.tolist()],
        facecolor=FLOOR_COLOR, edgecolor=FLOOR_EDGE, alpha=1.0, linewidth=0.8))

    # Floor grid
    x0, x1 = pts_m[:, 0].min(), pts_m[:, 0].max()
    y0, y1 = pts_m[:, 1].min(), pts_m[:, 1].max()
    grid_lines = []
    for gx in np.arange(x0, x1 + 0.5, 0.5):
        grid_lines.append([(gx, y0, 0.001), (gx, y1, 0.001)])
    for gy in np.arange(y0, y1 + 0.5, 0.5):
        grid_lines.append([(x0, gy, 0.001), (x1, gy, 0.001)])
    ax.add_collection3d(Line3DCollection(grid_lines, colors=FLOOR_EDGE,
                                         linewidths=0.4, alpha=0.5))

    # Drop shadow
    shadow_pts = np.column_stack([pts_m[:, 0] + 0.04, pts_m[:, 1] + 0.04,
                                   np.full(n, -0.005)])
    ax.add_collection3d(Poly3DCollection([shadow_pts.tolist()],
        facecolor="#aaaaaa", edgecolor="none", alpha=0.25, linewidth=0))

    # Back walls
    for i, quad in back_walls:
        fc, ec, alpha = wall_shade(wall_normals[i], False)
        ax.add_collection3d(Poly3DCollection([quad], facecolor=fc, edgecolor=ec,
                                              alpha=alpha, linewidth=0.7))

    # Floor-wall base AO
    for i, quad in back_walls:
        p0 = np.array(quad[0]); p1 = np.array(quad[1])
        c3d = np.array([centroid[0], centroid[1], 0.0])
        si = 0.12
        s0 = p0 + (c3d - p0) * si; s0[2] = 0.002
        s1 = p1 + (c3d - p1) * si; s1[2] = 0.002
        p0c = p0.copy(); p0c[2] = 0.002
        p1c = p1.copy(); p1c[2] = 0.002
        ax.add_collection3d(Poly3DCollection([[p0c, p1c, s1, s0]],
            facecolor="#888888", edgecolor="none", alpha=0.18, linewidth=0))

    # Vertical corner AO strips
    AO_STEPS = 10; AO_WIDTH = 0.30; AO_ALPHA = 0.07
    for i in range(n):
        cf = floor_pts[i].copy(); cc = ceiling_pts[i].copy()
        inward = np.array([centroid[0] - cf[0], centroid[1] - cf[1], 0.0])
        inward /= (np.linalg.norm(inward) + 1e-9)
        for side_dir_raw in [floor_pts[(i-1)%n] - cf, floor_pts[(i+1)%n] - cf]:
            sd = side_dir_raw.copy(); sd[2] = 0
            sd /= (np.linalg.norm(sd) + 1e-9)
            for step in range(AO_STEPS):
                t0 = (step / AO_STEPS) * AO_WIDTH
                t1 = ((step+1) / AO_STEPS) * AO_WIDTH
                fade = ((1 - step / AO_STEPS) ** 2) * AO_ALPHA
                p0f = cf + sd * t0; p0f[2] = 0.004
                p1f = cf + sd * t1; p1f[2] = 0.004
                p0c2 = cc + sd * t0
                p1c2 = cc + sd * t1
                ax.add_collection3d(Poly3DCollection([[p0f, p1f, p1c2, p0c2]],
                    facecolor="#222222", edgecolor="none",
                    alpha=fade, linewidth=0, zorder=5))

    # Floor-wall diffuse AO (all walls)
    AO_F_STEPS = 10; AO_F_WIDTH = 0.55; AO_F_ALPHA = 0.07
    for i, quad in (back_walls + front_walls):
        p0 = np.array(quad[0]); p1 = np.array(quad[1])
        c3d = np.array([centroid[0], centroid[1], 0.0])
        for step in range(AO_F_STEPS):
            t0 = (step / AO_F_STEPS) * AO_F_WIDTH
            t1 = ((step+1) / AO_F_STEPS) * AO_F_WIDTH
            fade = ((1 - step / AO_F_STEPS) ** 2) * AO_F_ALPHA
            def ins(p, t):
                v = c3d - p; v[2] = 0; v /= (np.linalg.norm(v) + 1e-9)
                r = p + v * t; r[2] = 0.004; return r
            s0a = ins(p0.copy(), t0); s1a = ins(p1.copy(), t0)
            s0b = ins(p0.copy(), t1); s1b = ins(p1.copy(), t1)
            ax.add_collection3d(Poly3DCollection([[s0a, s1a, s1b, s0b]],
                facecolor="#111111", edgecolor="none",
                alpha=fade, linewidth=0, zorder=4))

    # Front walls (transparent)
    for i, quad in front_walls:
        fc, ec, alpha = wall_shade(wall_normals[i], True)
        ax.add_collection3d(Poly3DCollection([quad], facecolor=fc, edgecolor=ec,
                                              alpha=alpha, linewidth=1.0, zorder=10))

    # Top + vertical edges
    top_edges  = [[ceiling_pts[i], ceiling_pts[(i+1)%n]] for i in range(n)]
    vert_edges = [[floor_pts[i], ceiling_pts[i]] for i in range(n)]
    ax.add_collection3d(Line3DCollection(top_edges,  colors="#555566", linewidths=1.2, alpha=0.9,  zorder=11))
    ax.add_collection3d(Line3DCollection(vert_edges, colors="#555566", linewidths=1.0, alpha=0.85, zorder=11))

    # ── 6. OCR LABELS ─────────────────────────────────────────────────────────
    try:
        big   = cv2.resize(img_bgr, (w_img*3, h_img*3), interpolation=cv2.INTER_CUBIC)
        data  = pytesseract.image_to_data(big, config="--psm 11",
                                           output_type=pytesseract.Output.DICT)
        tokens = []
        for idx, t in enumerate(data["text"]):
            t = t.strip()
            if not t: continue
            cx = (data["left"][idx] + data["width"][idx]  // 2) // 3
            cy = (data["top"][idx]  + data["height"][idx] // 2) // 3
            tokens.append((t, cx, cy))

        labels, skip = [], set()
        for idx, (t, cx, cy) in enumerate(tokens):
            if idx in skip: continue
            m = re.match(r"(\d+[.,]\d+)\s*m$", t)
            if m:
                labels.append((f"{m.group(1)} m", cx, cy)); continue
            if re.match(r"\d+[.,]\d+$", t) and idx+1 < len(tokens):
                nt, nx, ny = tokens[idx+1]
                if nt == "m" and abs(nx - cx) < 80:
                    labels.append((f"{t} m", (cx+nx)//2, (cy+ny)//2))
                    skip.add(idx+1)

        img_mids = [((best_points[i,0]+best_points[(i+1)%n,0])/2,
                     (best_points[i,1]+best_points[(i+1)%n,1])/2)
                    for i in range(n)]
        wall_labels, used = {}, set()
        for lstr, lx, ly in labels:
            best_d, best_w = 1e9, -1
            for wi, (mx, my) in enumerate(img_mids):
                if wi in used: continue
                d = (lx-mx)**2 + (ly-my)**2
                if d < best_d: best_d, best_w = d, wi
            if best_w >= 0:
                wall_labels[best_w] = lstr; used.add(best_w)

        front_indices = {i for i, _ in front_walls}
        for wi, lstr in wall_labels.items():
            j       = (wi + 1) % n
            mid_top = (ceiling_pts[wi] + ceiling_pts[j]) / 2
            is_front = wi in front_indices
            push    = 0.55 if is_front else 0.25
            out     = wall_normals[wi] * push
            ax.text(mid_top[0]+out[0], mid_top[1]+out[1],
                    mid_top[2] + (0.22 if is_front else 0.10),
                    lstr, fontsize=7.5, fontweight="bold", color="#222222",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#aaaaaa", alpha=0.95, linewidth=0.6),
                    zorder=50)
    except Exception as e:
        print(f"OCR skipped: {e}")

    # ── 7. CAMERA ─────────────────────────────────────────────────────────────
    pad = 0.3
    ax.set_xlim(pts_m[:,0].min()-pad, pts_m[:,0].max()+pad)
    ax.set_ylim(pts_m[:,1].min()-pad, pts_m[:,1].max()+pad)
    ax.set_zlim(-0.1, wall_h+pad)
    ax.set_box_aspect([span_x, span_y, wall_h])
    ax.view_init(elev=CAM_ELEV, azim=CAM_AZIM)

    # ── 8. EXPORT TO BYTES ────────────────────────────────────────────────────
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
