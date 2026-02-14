"""
process.py - UNIVERSAL polygon-to-isometric converter
Handles: convex, concave, L-shaped, U-shaped, notched, any vertex count.
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

# ── helpers ──

def _edge_len(pts, i, j=None):
    if j is None:
        j = (i + 1) % len(pts)
    return np.hypot(pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1])


def _polygon_is_valid(pts):
    """Accept any polygon ≥3 vertices with no degenerate micro-edges."""
    n = len(pts)
    if n < 3:
        return False
    perimeter = sum(_edge_len(pts, i) for i in range(n))
    if perimeter < 1e-6:
        return False
    min_edge = perimeter * 0.008          # 0.8 % – very permissive for small features
    return all(_edge_len(pts, i) >= min_edge for i in range(n))


def _remove_collinear(pts, angle_thresh_deg=6):
    """Remove vertices that lie (nearly) on a straight line between neighbours."""
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
        if abs(angle - 180) > angle_thresh_deg:   # keep real corners
            out.append(p1)
    return np.array(out) if len(out) >= 3 else pts


def _detect_polygon(contour):
    """Robust polygon detection – preserves both large concavities and small notches."""
    perimeter = cv2.arcLength(contour, True)
    contour_area = cv2.contourArea(contour)
    if contour_area < 1:
        return contour.reshape(-1, 2).astype(float)

    # Compute Hausdorff-like max deviation for each candidate
    contour_pts = contour.reshape(-1, 2).astype(float)

    def max_deviation(approx_pts):
        """Max distance from any contour point to the approximated polygon."""
        # For each contour point, find min distance to nearest approx edge
        n_a = len(approx_pts)
        max_d = 0
        # Sample contour points (use up to 200 for speed)
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
        # Score: lower deviation is better, small penalty for complexity
        # Normalize deviation by image diagonal for scale invariance
        score = -dev + n_verts * (-0.5)  # both are penalties, pick least total penalty
        candidates.append((dev, n_verts, score, pts))

    if not candidates:
        pts = contour.reshape(-1, 2).astype(float)
        pts = _remove_collinear(pts, angle_thresh_deg=5)
        return pts

    # Strategy: find the "elbow" – the simplest polygon that has low deviation
    # Sort by vertex count, then find where deviation drops significantly
    candidates.sort(key=lambda x: x[1])  # sort by vertex count
    
    # Find the minimum deviation achievable
    min_dev = min(c[0] for c in candidates)
    # Accept any candidate whose deviation is within 2x of minimum
    acceptable = [c for c in candidates if c[0] < min_dev * 2.0 + 5.0]
    # Among acceptable, prefer fewest vertices
    acceptable.sort(key=lambda x: x[1])
    
    return acceptable[0][3]


def _signed_area(pts_2d):
    """Shoelace signed area (positive = CCW)."""
    n = len(pts_2d)
    s = sum(pts_2d[i, 0] * pts_2d[(i+1)%n, 1] - pts_2d[(i+1)%n, 0] * pts_2d[i, 1] for i in range(n))
    return s / 2.0


def _ensure_ccw(pts_2d):
    if _signed_area(pts_2d) < 0:
        return pts_2d[::-1].copy()
    return pts_2d.copy()


def _outward_normal_2d(pts, i):
    """Outward unit normal for edge i→i+1 assuming CCW winding."""
    n = len(pts)
    j = (i + 1) % n
    dx = pts[j, 0] - pts[i, 0]
    dy = pts[j, 1] - pts[i, 1]
    # CCW winding → outward normal is (dy, -dx)
    nx, ny = dy, -dx
    length = math.hypot(nx, ny)
    if length < 1e-12:
        return np.array([0.0, 0.0])
    return np.array([nx / length, ny / length])


# ── OCR ──

def _ocr_labels(img_bgr, w_img, h_img):
    """Extract (value_m, cx, cy) labels from image via OCR."""
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
            # "3.19 m" in one token
            m = re.match(r"(\d+[.,]\d+)\s*[mM]$", t)
            if m:
                labels.append((float(m.group(1).replace(",", ".")), cx, cy))
                continue
            # number then separate "m"
            if re.match(r"\d+[.,]\d+$", t) and i + 1 < len(tokens):
                nt, nx, ny = tokens[i + 1]
                if re.match(r"[mM]$", nt) and abs(nx - cx) < 100:
                    labels.append((float(t.replace(",", ".")), (cx + nx) // 2, (cy + ny) // 2))
                    skip.add(i + 1)
                    continue
            # bare number (no degree symbol nearby)
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
    """Greedily assign OCR labels to nearest edges."""
    n = len(pts)
    edge_px = [_edge_len(pts, i) for i in range(n)]

    assignments = {}   # edge_idx → (value_m, label_str)
    used = set()

    # build (distance, edge_idx, label_idx) triples, sort by distance
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
    # grid clipped to polygon
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

    # ── 9. walls – depth-sorted ──
    wall_quads = []
    for i in range(n):
        j = (i + 1) % n
        quad = [floor_3d[i], floor_3d[j], ceiling_3d[j], ceiling_3d[i]]
        normal_2d = _outward_normal_2d(pts_m, i)
        normal_3d = np.array([normal_2d[0], normal_2d[1], 0.0])
        dot = np.dot(normal_3d, cam_dir)
        wall_quads.append((i, quad, normal_3d, dot))

    # draw back walls first (dot < 0 → facing away), then front walls translucent
    back  = [(i, q) for i, q, n3, d in wall_quads if d <= 0]
    front = [(i, q) for i, q, n3, d in wall_quads if d > 0]

    for i, q in back:
        ax.add_collection3d(Poly3DCollection([q], facecolor=WALL_DARK, edgecolor=EDGE_CLR,
                                             alpha=1.0, linewidth=0.8))
    for i, q in front:
        ax.add_collection3d(Poly3DCollection([q], facecolor=WALL_LIGHT, edgecolor=EDGE_CLR,
                                             alpha=0.45, linewidth=1.0, zorder=10))

    # top edges + vertical edges
    for i in range(n):
        j = (i + 1) % n
        ax.plot([ceiling_3d[i,0], ceiling_3d[j,0]], [ceiling_3d[i,1], ceiling_3d[j,1]],
                [ceiling_3d[i,2], ceiling_3d[j,2]], color=EDGE_CLR, linewidth=1.2, alpha=0.9)
        ax.plot([floor_3d[i,0], ceiling_3d[i,0]], [floor_3d[i,1], ceiling_3d[i,1]],
                [floor_3d[i,2], ceiling_3d[i,2]], color=EDGE_CLR, linewidth=1.0, alpha=0.85)

    # ── 10. labels ──
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

    # ── 11. axes limits ──
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
