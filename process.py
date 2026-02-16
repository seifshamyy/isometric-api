"""
process.py - UNIVERSAL polygon-to-isometric converter
Handles: convex, concave, L-shaped, U-shaped, notched, any vertex count.
NOW WITH: colored segment detection for fixtures/openings.
"""

import io, re, math
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pytesseract
from matplotlib.path import Path as MplPath

# ── rendering constants ──
WALL_H       = 2.8
CAM_ELEV     = 48
CAM_AZIM     = -45
BG           = "#dde0e5"
FLOOR_CLR    = "#c8c8c8"
FLOOR_EDGE   = "#b8b8b8"
WALL_LIGHT   = "#909090"
WALL_DARK    = "#7a7a7a"
EDGE_CLR     = "#555566"
LABEL_MAX_D  = 180        # px – max label-to-edge distance

# ── fixture standard heights (metres) ──
SINK_H       = 0.80       # 0.8m high as specified
SINK_DEPTH   = 0.55       # how far sink sticks out from wall
FRIDGE_H     = 1.80       # standard fridge height
FRIDGE_DEPTH = 0.65       # depth from wall
OVEN_H       = 0.85       # same as counter
OVEN_DEPTH   = 0.55
WINDOW_BOTTOM = 0.90      # sill height from floor
WINDOW_TOP    = 2.10      # top of window from floor
WINDOW_DEPTH  = 0.08      # thin recess
DOOR_H        = 2.10      # standard door height
DOOR_DEPTH    = 0.10      # thin frame

# ── fixture colors for 3D rendering (match line colors) ──
SINK_CLR      = "#FF69B4"   # pink (matches pink line)
SINK_EDGE     = "#CC5590"
FRIDGE_CLR    = "#CC00CC"   # magenta (matches magenta line)
FRIDGE_EDGE   = "#990099"
OVEN_CLR      = "#E03030"   # red (matches red line)
OVEN_EDGE     = "#AA2020"
WINDOW_CLR    = "#FFE44D"   # yellow (matches yellow line)
WINDOW_EDGE   = "#CCA800"
DOOR_CLR      = "#D2B48C"   # beige/tan
DOOR_EDGE     = "#8a6d4a"
OPENING_CLR   = FLOOR_CLR

# ── color detection ranges (HSV) ──
# Each entry: list of (lower_hsv, upper_hsv) ranges
COLOR_RANGES = {
    "green":   [((35, 60, 60),   (85, 255, 255))],
    "pink":    [((160, 60, 100), (175, 255, 255))],    # H 160-175 = pink/hot pink
    "magenta": [((145, 60, 50),  (159, 255, 255))],    # H 145-159 = magenta/purple-pink
    "yellow":  [((15, 60, 80),   (35, 255, 255))],
    "red":     [((0, 100, 80),   (10, 255, 255)),
                ((170, 100, 80), (180, 255, 255))],
    "blue":    [((90, 60, 80),   (135, 255, 255))],
}

COLOR_TO_FIXTURE = {
    "green":   "opening",
    "pink":    "sink",
    "magenta": "fridge",
    "yellow":  "window",
    "red":     "stove",
    "blue":    "door",
}


# ── helpers ──

def _edge_len(pts, i, j=None):
    if j is None:
        j = (i + 1) % len(pts)
    return np.hypot(pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1])


def _polygon_is_valid(pts):
    n = len(pts)
    if n < 3:
        return False
    perimeter = sum(_edge_len(pts, i) for i in range(n))
    if perimeter < 1e-6:
        return False
    min_edge = perimeter * 0.008
    return all(_edge_len(pts, i) >= min_edge for i in range(n))


def _remove_collinear(pts, angle_thresh_deg=6):
    out = []
    n = len(pts)
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        v1 = p0 - p1
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        angle = math.degrees(math.acos(np.clip(cos_a, -1, 1)))
        if abs(angle - 180) > angle_thresh_deg:
            out.append(p1)
    return np.array(out) if len(out) >= 3 else pts


def _detect_polygon(contour):
    perimeter = cv2.arcLength(contour, True)
    contour_area = cv2.contourArea(contour)
    if contour_area < 1:
        return contour.reshape(-1, 2).astype(float)

    contour_pts = contour.reshape(-1, 2).astype(float)

    def max_deviation(approx_pts):
        n_a = len(approx_pts)
        max_d = 0
        step = max(1, len(contour_pts) // 200)
        for cp in contour_pts[::step]:
            min_d = float('inf')
            for i in range(n_a):
                j = (i + 1) % n_a
                d = _perp_dist(cp[0], cp[1], approx_pts[i], approx_pts[j])
                min_d = min(min_d, d)
            max_d = max(max_d, min_d)
        return max_d

    candidates = []
    for eps_mult in np.linspace(0.002, 0.06, 50):
        approx = cv2.approxPolyDP(contour, eps_mult * perimeter, True)
        pts = approx.reshape(-1, 2).astype(float)
        pts = _remove_collinear(pts)
        if not _polygon_is_valid(pts):
            continue
        dev = max_deviation(pts)
        n_verts = len(pts)
        score = -dev + n_verts * (-0.5)
        candidates.append((dev, n_verts, score, pts))

    if not candidates:
        pts = contour.reshape(-1, 2).astype(float)
        pts = _remove_collinear(pts, angle_thresh_deg=5)
        return pts

    candidates.sort(key=lambda x: x[1])
    min_dev = min(c[0] for c in candidates)
    acceptable = [c for c in candidates if c[0] < min_dev * 2.0 + 5.0]
    acceptable.sort(key=lambda x: x[1])
    return acceptable[0][3]


def _signed_area(pts_2d):
    n = len(pts_2d)
    s = sum(pts_2d[i, 0] * pts_2d[(i+1)%n, 1] - pts_2d[(i+1)%n, 0] * pts_2d[i, 1] for i in range(n))
    return s / 2.0


def _ensure_ccw(pts_2d):
    if _signed_area(pts_2d) < 0:
        return pts_2d[::-1].copy()
    return pts_2d.copy()


def _outward_normal_2d(pts, i):
    n = len(pts)
    j = (i + 1) % n
    dx = pts[j, 0] - pts[i, 0]
    dy = pts[j, 1] - pts[i, 1]
    nx, ny = dy, -dx
    length = math.hypot(nx, ny)
    if length < 1e-12:
        return np.array([0.0, 0.0])
    return np.array([nx / length, ny / length])


def _inward_normal_2d(pts, i):
    return -_outward_normal_2d(pts, i)


# ── OCR ──

def _ocr_labels(img_bgr, w_img, h_img):
    try:
        scale = 3
        big = cv2.resize(img_bgr, (w_img * scale, h_img * scale), interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(big, config="--psm 11", output_type=pytesseract.Output.DICT)
        tokens = []
        for i, t in enumerate(data["text"]):
            t = t.strip()
            if t:
                cx = (data["left"][i] + data["width"][i] // 2) // scale
                cy = (data["top"][i] + data["height"][i] // 2) // scale
                tokens.append((t, cx, cy))

        labels, skip = [], set()
        for i, (t, cx, cy) in enumerate(tokens):
            if i in skip:
                continue
            m = re.match(r"(\d+[.,]\d+)\s*[mM]$", t)
            if m:
                labels.append((float(m.group(1).replace(",", ".")), cx, cy))
                continue
            if re.match(r"\d+[.,]\d+$", t) and i + 1 < len(tokens):
                nt, nx, ny = tokens[i + 1]
                if re.match(r"[mM]$", nt) and abs(nx - cx) < 100:
                    labels.append((float(t.replace(",", ".")), (cx + nx) // 2, (cy + ny) // 2))
                    skip.add(i + 1)
                    continue
            if re.match(r"\d+[.,]\d+$", t):
                if i + 1 < len(tokens) and re.match(r"[°]", tokens[i + 1][0]):
                    continue
                labels.append((float(t.replace(",", ".")), cx, cy))
        return labels
    except Exception:
        return []


def _perp_dist(px, py, p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return math.hypot(px - p0[0], py - p0[1])
    t = max(0, min(1, ((px - p0[0]) * dx + (py - p0[1]) * dy) / l2))
    return math.hypot(px - (p0[0] + t * dx), py - (p0[1] + t * dy))


def _assign_labels(pts, labels):
    n = len(pts)
    edge_px = [_edge_len(pts, i) for i in range(n)]
    assignments = {}
    used = set()
    triples = []
    for li, (val, lx, ly) in enumerate(labels):
        for ei in range(n):
            d = _perp_dist(lx, ly, pts[ei], pts[(ei + 1) % n])
            if d < LABEL_MAX_D:
                triples.append((d, ei, li))
    triples.sort()
    for d, ei, li in triples:
        if ei in assignments or li in used:
            continue
        val = labels[li][0]
        assignments[ei] = (val, f"{val:.2f} m")
        used.add(li)
    return assignments


# ──────────────────────────────────────────────────────────────────────────────
# COLOR SEGMENT DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _detect_colored_segments(img_bgr, pts_px):
    """
    For each color, find colored pixels near the polygon edges.
    Returns: list of dicts with keys:
        edge_idx, fixture_type, t_start, t_end (parametric 0..1 along edge)
    """
    h_img, w_img = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    n = len(pts_px)
    fixtures = []

    for color_name, ranges in COLOR_RANGES.items():
        # Build mask for this color
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for (lo, hi) in ranges:
            lo_arr = np.array(lo, dtype=np.uint8)
            hi_arr = np.array(hi, dtype=np.uint8)
            mask |= cv2.inRange(hsv, lo_arr, hi_arr)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find colored pixel locations
        colored_pts = np.argwhere(mask > 0)  # (row, col) = (y, x)
        if len(colored_pts) < 10:
            continue

        colored_xy = colored_pts[:, ::-1].astype(float)  # (x, y)

        # For each edge, find colored pixels within proximity
        PROXIMITY = 20  # pixels
        for ei in range(n):
            ej = (ei + 1) % n
            p0 = pts_px[ei]
            p1 = pts_px[ej]
            edge_vec = p1 - p0
            edge_len_px = np.linalg.norm(edge_vec)
            if edge_len_px < 5:
                continue

            edge_dir = edge_vec / edge_len_px

            # Vector from p0 to each colored pixel
            vecs = colored_xy - p0
            # Project onto edge direction → parametric t
            t_vals = vecs @ edge_dir
            # Perpendicular distance
            perp = np.abs(vecs[:, 0] * (-edge_dir[1]) + vecs[:, 1] * edge_dir[0])

            # Filter: within edge bounds and close to edge
            valid = (t_vals >= -5) & (t_vals <= edge_len_px + 5) & (perp < PROXIMITY)
            t_valid = t_vals[valid]

            if len(t_valid) < 5:
                continue

            # Parametric range of the colored segment
            t_start = max(0, np.percentile(t_valid, 2)) / edge_len_px
            t_end = min(edge_len_px, np.percentile(t_valid, 98)) / edge_len_px

            if (t_end - t_start) < 0.02:  # too small
                continue

            fixture_type = COLOR_TO_FIXTURE[color_name]
            fixtures.append({
                "edge_idx": ei,
                "fixture_type": fixture_type,
                "t_start": t_start,
                "t_end": t_end,
                "color_name": color_name,
            })

    return fixtures


# ──────────────────────────────────────────────────────────────────────────────
# 3D FIXTURE RENDERING
# ──────────────────────────────────────────────────────────────────────────────

def _render_fixtures(ax, fixtures, floor_3d, ceiling_3d, pts_m, scale, cam_dir):
    """Render all detected fixtures in the 3D scene."""
    n = len(pts_m)

    for fix in fixtures:
        ei = fix["edge_idx"]
        ej = (ei + 1) % n
        t0 = fix["t_start"]
        t1 = fix["t_end"]
        ftype = fix["fixture_type"]

        # 3D positions along the edge at floor level
        p_start_floor = floor_3d[ei] + t0 * (floor_3d[ej] - floor_3d[ei])
        p_end_floor   = floor_3d[ei] + t1 * (floor_3d[ej] - floor_3d[ei])

        # Inward normal (into the room)
        inward_2d = _inward_normal_2d(pts_m, ei)
        inward_3d = np.array([inward_2d[0], inward_2d[1], 0.0])

        # Outward normal
        outward_2d = _outward_normal_2d(pts_m, ei)
        outward_3d = np.array([outward_2d[0], outward_2d[1], 0.0])

        # Normal dot with camera for visibility
        normal_3d = outward_3d
        dot = np.dot(normal_3d, cam_dir)

        fixture_width = np.linalg.norm(p_end_floor - p_start_floor)

        if ftype == "opening":
            _render_opening(ax, p_start_floor, p_end_floor, WALL_H)

        elif ftype == "sink":
            _render_box_fixture(ax, p_start_floor, p_end_floor, inward_3d,
                                floor_z=0.0, bottom_z=0.0,
                                top_z=SINK_H, depth=SINK_DEPTH,
                                face_clr=SINK_CLR, edge_clr=SINK_EDGE,
                                label="Sink", dot=dot)

        elif ftype == "fridge":
            _render_box_fixture(ax, p_start_floor, p_end_floor, inward_3d,
                                floor_z=0.0, bottom_z=0.0,
                                top_z=FRIDGE_H, depth=FRIDGE_DEPTH,
                                face_clr=FRIDGE_CLR, edge_clr=FRIDGE_EDGE,
                                label="Fridge", dot=dot)

        elif ftype == "stove":
            _render_box_fixture(ax, p_start_floor, p_end_floor, inward_3d,
                                floor_z=0.0, bottom_z=0.0,
                                top_z=OVEN_H, depth=OVEN_DEPTH,
                                face_clr=OVEN_CLR, edge_clr=OVEN_EDGE,
                                label="Stove", dot=dot)

        elif ftype == "window":
            _render_window(ax, p_start_floor, p_end_floor, outward_3d,
                           label="Window", dot=dot)

        elif ftype == "door":
            _render_door(ax, p_start_floor, p_end_floor, inward_3d,
                         label="Door", dot=dot)


def _render_opening(ax, p_start, p_end, wall_h):
    """Opening = no wall. We do nothing here; the wall renderer will skip this segment."""
    # Openings are handled by modifying wall rendering, not by adding geometry.
    pass


def _render_box_fixture(ax, p_start, p_end, inward_3d,
                         floor_z, bottom_z, top_z, depth,
                         face_clr, edge_clr, label, dot):
    """Render a box (sink, fridge, oven) against the wall."""
    # Four base corners at bottom_z
    c0 = p_start.copy(); c0[2] = bottom_z
    c1 = p_end.copy();   c1[2] = bottom_z
    c2 = p_end   + inward_3d * depth; c2[2] = bottom_z
    c3 = p_start + inward_3d * depth; c3[2] = bottom_z

    # Four top corners at top_z
    c4 = c0.copy(); c4[2] = top_z
    c5 = c1.copy(); c5[2] = top_z
    c6 = c2.copy(); c6[2] = top_z
    c7 = c3.copy(); c7[2] = top_z

    faces = [
        [c0, c1, c5, c4],  # back face (against wall)
        [c2, c3, c7, c6],  # front face
        [c0, c3, c7, c4],  # left face
        [c1, c2, c6, c5],  # right face
        [c4, c5, c6, c7],  # top face
    ]
    if bottom_z > 0.01:
        faces.append([c0, c1, c2, c3])  # bottom face (visible if raised)

    alpha = 0.92 if dot > 0 else 1.0
    for face in faces:
        ax.add_collection3d(Poly3DCollection(
            [face], facecolor=face_clr, edgecolor=edge_clr,
            alpha=alpha, linewidth=0.8, zorder=20
        ))

    # Draw wireframe edges explicitly for visibility against z-fighting
    all_corners = [c0, c1, c2, c3, c4, c5, c6, c7]
    edges_wire = [
        (c0, c1), (c1, c2), (c2, c3), (c3, c0),  # bottom
        (c4, c5), (c5, c6), (c6, c7), (c7, c4),  # top
        (c0, c4), (c1, c5), (c2, c6), (c3, c7),  # verticals
    ]
    for a, b in edges_wire:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=edge_clr, linewidth=1.0, alpha=0.9, zorder=25)

    # Label on top
    center_top = (c4 + c5 + c6 + c7) / 4.0
    ax.text(center_top[0], center_top[1], center_top[2] + 0.08,
            label, fontsize=6.5, fontweight="bold", color="#111",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#999", alpha=0.9, linewidth=0.5),
            zorder=55)


def _render_window(ax, p_start, p_end, outward_3d, label, dot):
    """Render a window as a recessed glass panel in the wall."""
    # Window frame corners
    c0 = p_start.copy(); c0[2] = WINDOW_BOTTOM
    c1 = p_end.copy();   c1[2] = WINDOW_BOTTOM
    c2 = p_end.copy();   c2[2] = WINDOW_TOP
    c3 = p_start.copy(); c3[2] = WINDOW_TOP

    # Glass pane (slightly recessed outward for visibility)
    offset = outward_3d * 0.02
    glass = [c0 + offset, c1 + offset, c2 + offset, c3 + offset]

    alpha = 0.5 if dot > 0 else 0.7
    # Glass
    ax.add_collection3d(Poly3DCollection(
        [glass], facecolor=WINDOW_CLR, edgecolor=WINDOW_EDGE,
        alpha=alpha, linewidth=1.0, zorder=12
    ))
    # Frame lines
    for a, b in [(c0, c1), (c1, c2), (c2, c3), (c3, c0)]:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=WINDOW_EDGE, linewidth=1.5, alpha=0.9, zorder=13)
    # Cross bars
    mid_h = (WINDOW_BOTTOM + WINDOW_TOP) / 2
    mid_left = p_start.copy(); mid_left[2] = mid_h
    mid_right = p_end.copy(); mid_right[2] = mid_h
    ax.plot([mid_left[0], mid_right[0]], [mid_left[1], mid_right[1]],
            [mid_left[2], mid_right[2]], color=WINDOW_EDGE, linewidth=1.0, alpha=0.7, zorder=13)
    mid_w = (p_start + p_end) / 2
    mid_bot = mid_w.copy(); mid_bot[2] = WINDOW_BOTTOM
    mid_top = mid_w.copy(); mid_top[2] = WINDOW_TOP
    ax.plot([mid_bot[0], mid_top[0]], [mid_bot[1], mid_top[1]],
            [mid_bot[2], mid_top[2]], color=WINDOW_EDGE, linewidth=1.0, alpha=0.7, zorder=13)

    # Label
    center = (c2 + c3) / 2.0
    ax.text(center[0], center[1], center[2] + 0.12,
            label, fontsize=6.5, fontweight="bold", color="#111",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF8B0",
                      edgecolor=WINDOW_EDGE, alpha=0.9, linewidth=0.5),
            zorder=55)


def _render_door(ax, p_start, p_end, inward_3d, label, dot):
    """Render a door as a cutout with a thin frame and arc suggestion."""
    # Door frame
    c0 = p_start.copy(); c0[2] = 0.0
    c1 = p_end.copy();   c1[2] = 0.0
    c2 = p_end.copy();   c2[2] = DOOR_H
    c3 = p_start.copy(); c3[2] = DOOR_H

    # Frame (thin rectangle slightly inside room)
    offset = inward_3d * 0.03
    frame = [c0 + offset, c1 + offset, c2 + offset, c3 + offset]

    alpha = 0.35 if dot > 0 else 0.55
    ax.add_collection3d(Poly3DCollection(
        [frame], facecolor=DOOR_CLR, edgecolor=DOOR_EDGE,
        alpha=alpha, linewidth=1.0, zorder=12
    ))

    # Door frame lines
    for a, b in [(c0, c1), (c1, c2), (c2, c3), (c3, c0)]:
        ap = a + offset
        bp = b + offset
        ax.plot([ap[0], bp[0]], [ap[1], bp[1]], [ap[2], bp[2]],
                color=DOOR_EDGE, linewidth=1.5, alpha=0.8, zorder=13)

    # Swing arc (quarter circle on floor)
    door_width = np.linalg.norm(p_end[:2] - p_start[:2])
    arc_center = p_start[:2].copy()
    door_dir = (p_end[:2] - p_start[:2])
    door_dir_norm = door_dir / (np.linalg.norm(door_dir) + 1e-12)
    swing_dir = inward_3d[:2]

    n_arc = 20
    arc_x, arc_y = [], []
    for k in range(n_arc + 1):
        frac = k / n_arc
        angle = frac * math.pi / 2
        pt = arc_center + door_width * (math.cos(angle) * door_dir_norm + math.sin(angle) * swing_dir)
        arc_x.append(pt[0])
        arc_y.append(pt[1])
    ax.plot(arc_x, arc_y, [0.005] * len(arc_x),
            color=DOOR_EDGE, linewidth=1.0, alpha=0.5, linestyle="--", zorder=5)

    # Label
    center = (c2 + c3) / 2.0
    ax.text(center[0], center[1], center[2] + 0.12,
            label, fontsize=6.5, fontweight="bold", color="#111",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=DOOR_CLR,
                      edgecolor=DOOR_EDGE, alpha=0.9, linewidth=0.5),
            zorder=55)


# ──────────────────────────────────────────────────────────────────────────────
# WALL RENDERING WITH OPENINGS
# ──────────────────────────────────────────────────────────────────────────────

def _get_wall_segments(edge_idx, fixtures, n_pts):
    """
    Given an edge, return list of (t_start, t_end, is_solid) segments.
    Fixtures that are openings, windows, or doors create gaps in the wall.
    """
    # Collect gap ranges for this edge
    gaps = []
    for fix in fixtures:
        if fix["edge_idx"] != edge_idx:
            continue
        ftype = fix["fixture_type"]
        if ftype == "opening":
            gaps.append((fix["t_start"], fix["t_end"]))
        elif ftype == "door":
            gaps.append((fix["t_start"], fix["t_end"]))
        elif ftype == "window":
            # Window doesn't fully remove wall, just the middle part
            # We'll render the wall with a hole for the window
            gaps.append((fix["t_start"], fix["t_end"]))

    if not gaps:
        return [(0.0, 1.0, True, None)]

    # Sort and merge overlapping gaps
    gaps.sort()
    merged = [gaps[0]]
    for g in gaps[1:]:
        if g[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], g[1]))
        else:
            merged.append(g)

    # Build segment list
    segments = []
    cursor = 0.0
    for g_start, g_end in merged:
        if cursor < g_start - 0.001:
            segments.append((cursor, g_start, True, None))
        # Find what fixture created this gap
        gap_fixture = None
        for fix in fixtures:
            if fix["edge_idx"] == edge_idx and abs(fix["t_start"] - g_start) < 0.01:
                gap_fixture = fix
                break
        segments.append((g_start, g_end, False, gap_fixture))
        cursor = g_end
    if cursor < 0.999:
        segments.append((cursor, 1.0, True, None))

    return segments


def _render_wall_segment(ax, floor_3d, ceiling_3d, ei, ej, t0, t1, is_front, fixture=None):
    """Render a wall segment, possibly with window cutout."""
    p0_f = floor_3d[ei] + t0 * (floor_3d[ej] - floor_3d[ei])
    p1_f = floor_3d[ei] + t1 * (floor_3d[ej] - floor_3d[ei])
    p0_c = ceiling_3d[ei] + t0 * (ceiling_3d[ej] - ceiling_3d[ei])
    p1_c = ceiling_3d[ei] + t1 * (ceiling_3d[ej] - ceiling_3d[ei])

    quad = [p0_f, p1_f, p1_c, p0_c]

    if is_front:
        clr = WALL_LIGHT
        alpha = 0.45
        lw = 1.0
        zorder = 10
    else:
        clr = WALL_DARK
        alpha = 1.0
        lw = 0.8
        zorder = 1

    ax.add_collection3d(Poly3DCollection(
        [quad], facecolor=clr, edgecolor=EDGE_CLR,
        alpha=alpha, linewidth=lw, zorder=zorder
    ))


def _render_wall_with_window_hole(ax, floor_3d, ceiling_3d, ei, ej, t0, t1, is_front):
    """
    Render wall around a window cutout.
    Below sill, above lintel, and to the sides of the window.
    """
    p0_f = floor_3d[ei] + t0 * (floor_3d[ej] - floor_3d[ei])
    p1_f = floor_3d[ei] + t1 * (floor_3d[ej] - floor_3d[ei])
    p0_c = ceiling_3d[ei] + t0 * (ceiling_3d[ej] - ceiling_3d[ei])
    p1_c = ceiling_3d[ei] + t1 * (ceiling_3d[ej] - ceiling_3d[ei])

    clr = WALL_LIGHT if is_front else WALL_DARK
    alpha = 0.45 if is_front else 1.0
    lw = 1.0 if is_front else 0.8
    zorder = 10 if is_front else 1

    # Below sill
    p0_sill = p0_f.copy(); p0_sill[2] = WINDOW_BOTTOM
    p1_sill = p1_f.copy(); p1_sill[2] = WINDOW_BOTTOM
    below = [p0_f, p1_f, p1_sill, p0_sill]
    ax.add_collection3d(Poly3DCollection(
        [below], facecolor=clr, edgecolor=EDGE_CLR,
        alpha=alpha, linewidth=lw, zorder=zorder
    ))

    # Above lintel
    p0_lint = p0_f.copy(); p0_lint[2] = WINDOW_TOP
    p1_lint = p1_f.copy(); p1_lint[2] = WINDOW_TOP
    above = [p0_lint, p1_lint, p1_c, p0_c]
    ax.add_collection3d(Poly3DCollection(
        [above], facecolor=clr, edgecolor=EDGE_CLR,
        alpha=alpha, linewidth=lw, zorder=zorder
    ))


def _render_wall_with_door_hole(ax, floor_3d, ceiling_3d, ei, ej, t0, t1, is_front):
    """Render wall above a door opening."""
    p0_f = floor_3d[ei] + t0 * (floor_3d[ej] - floor_3d[ei])
    p1_f = floor_3d[ei] + t1 * (floor_3d[ej] - floor_3d[ei])
    p0_c = ceiling_3d[ei] + t0 * (ceiling_3d[ej] - ceiling_3d[ei])
    p1_c = ceiling_3d[ei] + t1 * (ceiling_3d[ej] - ceiling_3d[ei])

    clr = WALL_LIGHT if is_front else WALL_DARK
    alpha = 0.45 if is_front else 1.0
    lw = 1.0 if is_front else 0.8
    zorder = 10 if is_front else 1

    # Only render above door height
    p0_top = p0_f.copy(); p0_top[2] = DOOR_H
    p1_top = p1_f.copy(); p1_top[2] = DOOR_H
    above = [p0_top, p1_top, p1_c, p0_c]
    ax.add_collection3d(Poly3DCollection(
        [above], facecolor=clr, edgecolor=EDGE_CLR,
        alpha=alpha, linewidth=lw, zorder=zorder
    ))


# ── main entry ──

def generate_isometric(img_bgr: np.ndarray) -> bytes:
    h_img, w_img = img_bgr.shape[:2]

    # ── 1. find contour ──
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image")

    cnt = max(contours, key=cv2.contourArea)

    # ── 2. extract polygon ──
    pts_px = _detect_polygon(cnt)
    pts_px = _ensure_ccw(pts_px)
    n = len(pts_px)

    # ── 3. OCR labels ──
    labels = _ocr_labels(img_bgr, w_img, h_img)
    wall_meas = _assign_labels(pts_px, labels)

    # ── 3b. detect colored segments (fixtures) ──
    fixtures = _detect_colored_segments(img_bgr, pts_px)
    print(f"[DEBUG] Detected {len(fixtures)} fixture segments:")
    for f in fixtures:
        print(f"  edge {f['edge_idx']}: {f['fixture_type']} ({f['color_name']}) "
              f"t=[{f['t_start']:.3f}, {f['t_end']:.3f}]")

    # ── 4. compute scale (px → metres) ──
    edge_px = [_edge_len(pts_px, i) for i in range(n)]
    if wall_meas:
        scales = []
        for ei, (val_m, _) in wall_meas.items():
            if edge_px[ei] > 0:
                scales.append(val_m / edge_px[ei])
        scale = float(np.median(scales)) if scales else 4.0 / max(edge_px)
    else:
        scale = 4.0 / max(edge_px)

    # fill in unlabelled edges
    for i in range(n):
        if i not in wall_meas:
            m = edge_px[i] * scale
            wall_meas[i] = (m, f"{m:.2f} m")

    # ── 5. build 3D points ──
    pts_m = pts_px * scale
    pts_m[:, 0] -= pts_m[:, 0].min()
    pts_m[:, 1] -= pts_m[:, 1].min()

    floor_3d   = np.column_stack([pts_m[:, 0], pts_m[:, 1], np.zeros(n)])
    ceiling_3d = np.column_stack([pts_m[:, 0], pts_m[:, 1], np.full(n, WALL_H)])

    # ── 6. figure ──
    span_x = pts_m[:, 0].max() - pts_m[:, 0].min()
    span_y = pts_m[:, 1].max() - pts_m[:, 1].min()
    fig = plt.figure(figsize=(13, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    ax.grid(False)
    ax.set_axis_off()
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("none")

    # ── 7. floor ──
    ax.add_collection3d(
        Poly3DCollection([floor_3d.tolist()],
                         facecolor=FLOOR_CLR, edgecolor=FLOOR_EDGE, alpha=1, linewidth=0.8)
    )
    path_2d = MplPath(pts_m)
    for x in np.arange(0, span_x + 0.5, 0.5):
        segs = []
        for y in np.arange(0, span_y + 0.5, 0.05):
            if path_2d.contains_point((x, y)):
                segs.append((x, y))
        if len(segs) >= 2:
            ax.plot([segs[0][0], segs[-1][0]], [segs[0][1], segs[-1][1]],
                    [0.001, 0.001], color=FLOOR_EDGE, linewidth=0.35, alpha=0.45)
    for y in np.arange(0, span_y + 0.5, 0.5):
        segs = []
        for x in np.arange(0, span_x + 0.5, 0.05):
            if path_2d.contains_point((x, y)):
                segs.append((x, y))
        if len(segs) >= 2:
            ax.plot([segs[0][0], segs[-1][0]], [segs[0][1], segs[-1][1]],
                    [0.001, 0.001], color=FLOOR_EDGE, linewidth=0.35, alpha=0.45)

    # ── 8. camera direction for front/back sorting ──
    elev_r = np.radians(CAM_ELEV)
    azim_r = np.radians(CAM_AZIM)
    cam_dir = np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r),
    ])

    # ── 9. walls – with fixture-aware segmentation ──
    for i in range(n):
        j = (i + 1) % n
        normal_2d = _outward_normal_2d(pts_m, i)
        normal_3d = np.array([normal_2d[0], normal_2d[1], 0.0])
        dot = np.dot(normal_3d, cam_dir)
        is_front = dot > 0

        segments = _get_wall_segments(i, fixtures, n)

        for t0, t1, is_solid, fixture in segments:
            if is_solid:
                _render_wall_segment(ax, floor_3d, ceiling_3d, i, j, t0, t1, is_front)
            else:
                # Gap type depends on fixture
                if fixture is None:
                    continue  # pure opening
                ftype = fixture["fixture_type"]
                if ftype == "opening":
                    pass  # no wall at all
                elif ftype == "window":
                    _render_wall_with_window_hole(ax, floor_3d, ceiling_3d, i, j, t0, t1, is_front)
                elif ftype == "door":
                    _render_wall_with_door_hole(ax, floor_3d, ceiling_3d, i, j, t0, t1, is_front)

    # top edges + vertical edges
    for i in range(n):
        j = (i + 1) % n
        ax.plot([ceiling_3d[i,0], ceiling_3d[j,0]], [ceiling_3d[i,1], ceiling_3d[j,1]],
                [ceiling_3d[i,2], ceiling_3d[j,2]], color=EDGE_CLR, linewidth=1.2, alpha=0.9)
        ax.plot([floor_3d[i,0], ceiling_3d[i,0]], [floor_3d[i,1], ceiling_3d[i,1]],
                [floor_3d[i,2], ceiling_3d[i,2]], color=EDGE_CLR, linewidth=1.0, alpha=0.85)

    # ── 10. render fixtures (3D objects) ──
    _render_fixtures(ax, fixtures, floor_3d, ceiling_3d, pts_m, scale, cam_dir)

    # ── 11. labels ──
    centroid = pts_m.mean(axis=0)
    for i, (_, lbl) in wall_meas.items():
        j = (i + 1) % n
        mid_top = (ceiling_3d[i] + ceiling_3d[j]) / 2.0
        normal_2d = _outward_normal_2d(pts_m, i)
        offset = np.array([normal_2d[0], normal_2d[1], 0.0]) * 0.25
        ax.text(mid_top[0] + offset[0], mid_top[1] + offset[1], mid_top[2] + 0.15,
                lbl, fontsize=7.5, fontweight="bold", color="#222",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#aaa", alpha=0.95, linewidth=0.6),
                zorder=50)

    # ── 12. axes limits ──
    pad = 0.5
    ax.set_xlim(pts_m[:, 0].min() - pad, pts_m[:, 0].max() + pad)
    ax.set_ylim(pts_m[:, 1].min() - pad, pts_m[:, 1].max() + pad)
    ax.set_zlim(-0.3, WALL_H + pad)
    ax.set_box_aspect([max(span_x, 0.1), max(span_y, 0.1), WALL_H])
    ax.view_init(elev=CAM_ELEV, azim=CAM_AZIM)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
