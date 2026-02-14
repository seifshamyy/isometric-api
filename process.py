"""
process.py - BULLETPROOF polygon detection with validation
"""

import io, re
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pytesseract

WALL_H_RATIO = 0.5
CAM_ELEV, CAM_AZIM = 48, -45
BG, FLOOR, FLOOR_E = "#dde0e5", "#c8c8c8", "#c8c8c8"
TONE_R, TONE_L, EDGE_R, EDGE_L = "#909090", "#7a7a7a", "#707070", "#2a2a2a"
MAX_LABEL_DIST = 150
OPEN_SHAPE_THRESHOLD = 0.15

def validate_polygon(pts):
    """Returns True if polygon has no micro-edges"""
    n = len(pts)
    if n < 3 or n > 12:
        return False
    
    # Calculate perimeter
    perimeter = sum([np.hypot(pts[(i+1)%n,0] - pts[i,0], pts[(i+1)%n,1] - pts[i,1]) for i in range(n)])
    
    # Check each edge - reject if any edge is < 3% of total perimeter
    min_edge_length = perimeter * 0.03
    for i in range(n):
        j = (i + 1) % n
        edge_len = np.hypot(pts[j,0] - pts[i,0], pts[j,1] - pts[i,1])
        if edge_len < min_edge_length:
            return False
    
    return True

def detect_polygon_robust(contour, attempts=10):
    """Try multiple epsilon values and return first VALID polygon"""
    perimeter = cv2.arcLength(contour, True)
    
    # Try epsilon from 0.01 to 0.04 in small steps
    for eps_mult in np.linspace(0.01, 0.04, attempts):
        eps = eps_mult * perimeter
        approx = cv2.approxPolyDP(contour, eps, True)
        pts = approx.reshape(-1, 2).astype(float)
        
        if validate_polygon(pts):
            return pts
    
    # Fallback: use convex hull with medium epsilon
    hull = cv2.convexHull(contour)
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    return approx.reshape(-1, 2).astype(float)

def generate_isometric(img_bgr: np.ndarray) -> bytes:
    h_img, w_img = img_bgr.shape[:2]

    # Detect polygon
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")

    # Find largest contour
    cnt_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    best_cnt = cnt_sorted[0]
    
    # Use robust detection
    best_pts = detect_polygon_robust(best_cnt)
    n = len(best_pts)

    # Detect open vs closed shape
    area = cv2.contourArea(best_cnt)
    x, y, bw, bh = cv2.boundingRect(best_cnt)
    bbox_area = bw * bh
    area_ratio = area / (bbox_area + 1e-9)
    is_open_shape = area_ratio < OPEN_SHAPE_THRESHOLD

    # OCR
    try:
        big = cv2.resize(img_bgr, (w_img*3, h_img*3), interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(big, config="--psm 11", output_type=pytesseract.Output.DICT)
        tokens = [(t.strip(), (data["left"][i] + data["width"][i]//2)//3, (data["top"][i] + data["height"][i]//2)//3)
                  for i, t in enumerate(data["text"]) if t.strip()]

        labels, skip = [], set()
        for i, (t, cx, cy) in enumerate(tokens):
            if i in skip:
                continue
            m = re.match(r"(\d+[.,]\d+)\s*[mM]$", t)
            if m:
                labels.append((float(m.group(1).replace(",", ".")), cx, cy))
            elif re.match(r"\d+[.,]\d+$", t) and i+1 < len(tokens):
                nt, nx, ny = tokens[i+1]
                if re.match(r"[mM]$", nt) and abs(nx - cx) < 80:
                    labels.append((float(t.replace(",", ".")), (cx+nx)//2, (cy+ny)//2))
                    skip.add(i+1)
                elif not re.match(r"[°]", nt):
                    labels.append((float(t.replace(",", ".")), cx, cy))
    except Exception:
        labels = []

    def perp_dist(px, py, p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0:
            return np.hypot(px - x0, py - y0)
        t = max(0, min(1, ((px - x0) * dx + (py - y0) * dy) / (dx*dx + dy*dy)))
        proj_x, proj_y = x0 + t * dx, y0 + t * dy
        return np.hypot(px - proj_x, py - proj_y)

    edge_lens_px = [np.hypot(best_pts[(i+1)%n,0] - best_pts[i,0],
                             best_pts[(i+1)%n,1] - best_pts[i,1]) for i in range(n)]
    max_edge_px = max(edge_lens_px)

    wall_meas = {}
    used_labels = set()
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = best_pts[i], best_pts[j]
        edge_px = edge_lens_px[i]
        best_d, best_label_idx = 1e9, -1
        for label_idx, (meas_m, lx, ly) in enumerate(labels):
            if label_idx in used_labels:
                continue
            dist = perp_dist(lx, ly, p0, p1)
            if dist < MAX_LABEL_DIST and dist < best_d:
                best_d, best_label_idx = dist, label_idx
        if best_label_idx >= 0:
            meas_m, _, _ = labels[best_label_idx]
            if edge_px < max_edge_px * 0.4 and meas_m > 5.0:
                meas_str = f"{meas_m:.2f}"
                if meas_str[0] == '7':
                    meas_m = float('1' + meas_str[1:])
            wall_meas[i] = (meas_m, f"{meas_m:.2f} m")
            used_labels.add(best_label_idx)

    if is_open_shape:
        visible_edges = list(wall_meas.keys()) if wall_meas else [0, 1]
    else:
        visible_edges = list(range(n))

    if wall_meas:
        scales = []
        for i, (meas_m, _) in wall_meas.items():
            j = (i + 1) % n
            px_len = np.hypot(best_pts[j, 0] - best_pts[i, 0], best_pts[j, 1] - best_pts[i, 1])
            if px_len > 0:
                scales.append(meas_m / px_len)
        scale_mpx = np.median(scales) if scales else 4.0 / max_edge_px
    else:
        scale_mpx = 4.0 / max_edge_px

    for i in visible_edges:
        if i not in wall_meas:
            computed_m = edge_lens_px[i] * scale_mpx
            wall_meas[i] = (computed_m, f"{computed_m:.2f} m")

    pts_m = best_pts * scale_mpx
    pts_m[:, 0] -= pts_m[:, 0].min()
    pts_m[:, 1] -= pts_m[:, 1].min()

    wall_h = 2.8
    span_x = pts_m[:, 0].max()
    span_y = pts_m[:, 1].max()

    floor_pts = np.column_stack([pts_m[:, 0], pts_m[:, 1], np.zeros(n)])
    ceiling_pts = np.column_stack([pts_m[:, 0], pts_m[:, 1], np.full(n, wall_h)])

    wall_faces = []
    for i in visible_edges:
        j = (i + 1) % n
        wall_faces.append([floor_pts[i], floor_pts[j], ceiling_pts[j], ceiling_pts[i]])

    fig = plt.figure(figsize=(13, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    ax.grid(False)
    ax.set_axis_off()
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("none")

    if is_open_shape:
        ground_z = -wall_h * 0.6
        floor_width = 0.4
        for i in visible_edges:
            j = (i + 1) % n
            p0, p1 = floor_pts[i].copy(), floor_pts[j].copy()
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            nx, ny = -dy, dx
            length = np.hypot(nx, ny)
            if length > 0:
                nx, ny = nx / length, ny / length
            f0 = np.array([p0[0], p0[1], ground_z])
            f1 = np.array([p1[0], p1[1], ground_z])
            f2 = np.array([p1[0] + nx * floor_width, p1[1] + ny * floor_width, ground_z])
            f3 = np.array([p0[0] + nx * floor_width, p0[1] + ny * floor_width, ground_z])
            ax.add_collection3d(Poly3DCollection([[f0, f1, f2, f3]], 
                facecolor=FLOOR, edgecolor=FLOOR_E, alpha=0.7, linewidth=0.5))
            for t in np.linspace(0, 1, 11):
                gp0 = f0 + t * (f1 - f0)
                gp1 = f3 + t * (f2 - f3)
                ax.plot([gp0[0], gp1[0]], [gp0[1], gp1[1]], [gp0[2], gp1[2]], 
                        color=FLOOR_E, linewidth=0.3, alpha=0.6)
    else:
        ax.add_collection3d(Poly3DCollection([floor_pts.tolist()], facecolor=FLOOR, edgecolor=FLOOR_E, alpha=1, linewidth=0.8))
        grid = [[(x, 0, 0.001), (x, span_y, 0.001)] for x in np.arange(0, span_x+0.5, 0.5)]
        grid += [[(0, y, 0.001), (span_x, y, 0.001)] for y in np.arange(0, span_y+0.5, 0.5)]
        ax.add_collection3d(Line3DCollection(grid, colors=FLOOR_E, linewidths=0.4, alpha=0.5))

    elev_r, azim_r = np.radians(CAM_ELEV), np.radians(CAM_AZIM)
    cam_dir = np.array([np.cos(elev_r)*np.cos(azim_r), np.cos(elev_r)*np.sin(azim_r), np.sin(elev_r)])

    centroid = pts_m.mean(axis=0)
    wall_normals = []
    for i in visible_edges:
        j = (i + 1) % n
        dx, dy = pts_m[j, 0] - pts_m[i, 0], pts_m[j, 1] - pts_m[i, 1]
        nx, ny = dy, -dx
        mid = (pts_m[i] + pts_m[j]) / 2
        if nx * (mid[0] - centroid[0]) + ny * (mid[1] - centroid[1]) < 0:
            nx, ny = -nx, -ny
        length = np.hypot(nx, ny)
        if length > 0:
            wall_normals.append(np.array([nx, ny, 0]) / length)
        else:
            wall_normals.append(np.array([0, 0, 0]))

    if not is_open_shape:
        back_walls = []
        front_walls = []
        for idx, i in enumerate(visible_edges):
            quad = wall_faces[idx]
            normal = wall_normals[idx]
            if np.dot(normal, cam_dir) > 0:
                front_walls.append((idx, quad))
            else:
                back_walls.append((idx, quad))
        
        for idx, quad in back_walls:
            ax.add_collection3d(Poly3DCollection([quad], facecolor=TONE_L, edgecolor=EDGE_L, alpha=1.0, linewidth=0.8))
        
        for idx, quad in front_walls:
            ax.add_collection3d(Poly3DCollection([quad], facecolor=TONE_L, edgecolor=EDGE_L, alpha=0.5, linewidth=1.0, zorder=10))
    else:
        for idx, quad in enumerate(wall_faces):
            ax.add_collection3d(Poly3DCollection([quad], facecolor=TONE_L, edgecolor=EDGE_L, alpha=1.0, linewidth=0.8))

    for idx, i in enumerate(visible_edges):
        j = (i + 1) % n
        ax.plot([ceiling_pts[i, 0], ceiling_pts[j, 0]], [ceiling_pts[i, 1], ceiling_pts[j, 1]], 
                [ceiling_pts[i, 2], ceiling_pts[j, 2]], color="#555566", linewidth=1.2, alpha=0.9)
        ax.plot([floor_pts[i, 0], ceiling_pts[i, 0]], [floor_pts[i, 1], ceiling_pts[i, 1]], 
                [floor_pts[i, 2], ceiling_pts[i, 2]], color="#555566", linewidth=1.0, alpha=0.85)

    for i, (_, lbl) in wall_meas.items():
        j = (i + 1) % n
        mid = (ceiling_pts[i] + ceiling_pts[j]) / 2
        if is_open_shape:
            ax.text(mid[0], mid[1], mid[2] + 0.3, lbl, fontsize=7.5, fontweight="bold", color="#222", 
                    ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.25", facecolor="white", 
                    edgecolor="#aaa", alpha=0.95, linewidth=0.6), zorder=50)
        else:
            dx, dy = pts_m[j, 0] - pts_m[i, 0], pts_m[j, 1] - pts_m[i, 1]
            nx, ny = dy, -dx
            midpt = (pts_m[i] + pts_m[j]) / 2
            if nx * (midpt[0] - centroid[0]) + ny * (midpt[1] - centroid[1]) < 0:
                nx, ny = -nx, -ny
            length = np.hypot(nx, ny)
            if length > 0:
                normal = np.array([nx, ny, 0]) / length
                out = normal * 0.25
                ax.text(mid[0] + out[0], mid[1] + out[1], mid[2] + 0.15, lbl, fontsize=7.5, fontweight="bold", 
                        color="#222", ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.25", 
                        facecolor="white", edgecolor="#aaa", alpha=0.95, linewidth=0.6), zorder=50)

    pad = 0.5
    ax.set_xlim(pts_m[:, 0].min() - pad, pts_m[:, 0].max() + pad)
    ax.set_ylim(pts_m[:, 1].min() - pad, pts_m[:, 1].max() + pad)
    if is_open_shape:
        ax.set_zlim(-wall_h * 0.7, wall_h + pad)
    else:
        ax.set_zlim(-0.3, wall_h + pad)
    ax.set_box_aspect([span_x if not is_open_shape else pts_m[:, 0].max() - pts_m[:, 0].min(), 
                        span_y if not is_open_shape else pts_m[:, 1].max() - pts_m[:, 1].min(), 
                        wall_h * 1.5 if is_open_shape else wall_h])
    ax.view_init(elev=CAM_ELEV, azim=CAM_AZIM)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
