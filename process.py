"""
process.py - OCR-based isometric room extrusion
Usage: python process.py sketch.jpg
"""

import sys, os, re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pytesseract

DEFAULT_IMAGE, WALL_H_RATIO, APPROX_EPS = "sketch.jpg", 0.5, 0.02
CAM_ELEV, CAM_AZIM = 48, -30
BG, FLOOR, FLOOR_E = "#dde0e5", "#c8c8c8", "#c8c8c8"
TONE_R, TONE_L, EDGE_R, EDGE_L = "#909090", "#7a7a7a", "#707070", "#2a2a2a"
MAX_LABEL_DIST = 150

image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
if not os.path.exists(image_path): sys.exit(f"❌  {image_path}")

img_bgr = cv2.imread(image_path)
h_img, w_img = img_bgr.shape[:2]
print(f"📷  {image_path}")

# Detect polygon
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(thresh, kernel, iterations=2)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours: sys.exit("❌  No contours")

cnt_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
best_pts = None
for cnt in cnt_sorted[:5]:
    if cv2.contourArea(cnt) < h_img * w_img * 0.02: continue
    eps = APPROX_EPS * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if 3 <= len(approx) <= 12:
        best_pts = approx.reshape(-1, 2).astype(float)
        break

if best_pts is None:
    hull = cv2.convexHull(cnt_sorted[0])
    best_pts = cv2.approxPolyDP(hull, APPROX_EPS * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(float)

n = len(best_pts)
print(f"✅  {n}-corner polygon")

# OCR
big = cv2.resize(img_bgr, (w_img*3, h_img*3), interpolation=cv2.INTER_CUBIC)
data = pytesseract.image_to_data(big, config="--psm 11", output_type=pytesseract.Output.DICT)
tokens = [(t.strip(), (data["left"][i] + data["width"][i]//2)//3, (data["top"][i] + data["height"][i]//2)//3)
          for i, t in enumerate(data["text"]) if t.strip()]

labels, skip = [], set()
for i, (t, cx, cy) in enumerate(tokens):
    if i in skip: continue
    # Match "X.XX m" or "X.XX M" (capital M)
    m = re.match(r"(\d+[.,]\d+)\s*[mM]$", t)
    if m:
        labels.append((float(m.group(1).replace(",", ".")), cx, cy))
    elif re.match(r"\d+[.,]\d+$", t) and i+1 < len(tokens):
        nt, nx, ny = tokens[i+1]
        if re.match(r"[mM]$", nt) and abs(nx - cx) < 80:
            labels.append((float(t.replace(",", ".")), (cx+nx)//2, (cy+ny)//2))
            skip.add(i+1)
    # Try to recover partial measurements like "9m" → might be "1.89m"
    elif re.match(r"(\d+)[mM]$", t) and i > 0:
        prev_t, prev_cx, prev_cy = tokens[i-1]
        if re.match(r"\d+\.\d*$", prev_t) and abs(prev_cx - cx) < 100:
            combined = prev_t + t[:-1]
            if re.match(r"\d+\.\d+$", combined):
                labels.append((float(combined), (prev_cx + cx)//2, (prev_cy + cy)//2))
                skip.add(i-1)

print(f"📏  OCR: {[f'{m:.2f}m' for m,_,_ in labels]}")

# Match using perpendicular distance to edge LINE
def perp_dist(px, py, p0, p1):
    x0, y0 = p0; x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0: return np.hypot(px - x0, py - y0)
    t = max(0, min(1, ((px - x0) * dx + (py - y0) * dy) / (dx*dx + dy*dy)))
    proj_x, proj_y = x0 + t * dx, y0 + t * dy
    return np.hypot(px - proj_x, py - proj_y)

# Calculate edge pixel lengths for OCR validation
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
        if label_idx in used_labels: continue
        dist = perp_dist(lx, ly, p0, p1)
        if dist < MAX_LABEL_DIST and dist < best_d:
            best_d, best_label_idx = dist, label_idx
    if best_label_idx >= 0:
        meas_m, _, _ = labels[best_label_idx]
        
        # OCR FIX: Short edges with large measurements = "1" misread as "7"
        if edge_px < max_edge_px * 0.4 and meas_m > 5.0:
            meas_str = f"{meas_m:.2f}"
            if meas_str[0] == '7':
                meas_m = float('1' + meas_str[1:])
                print(f"   🔧 OCR fix: 7.XX → 1.XX (short edge)")
        
        wall_meas[i] = (meas_m, f"{meas_m:.2f} m")
        used_labels.add(best_label_idx)
        print(f"   Edge {i} ({edge_px:.0f}px) → {meas_m:.2f} m")

# Scale
if wall_meas:
    scales = []
    for i, (meas_m, _) in wall_meas.items():
        j = (i + 1) % n
        px_len = np.hypot(best_pts[j, 0] - best_pts[i, 0],
                          best_pts[j, 1] - best_pts[i, 1])
        scales.append(meas_m / px_len)
    scale_mpx = np.median(scales)
else:
    scale_mpx = 4.0 / max(edge_lens_px)

# Add computed measurements for edges without OCR labels
for i in range(n):
    if i not in wall_meas:
        computed_m = edge_lens_px[i] * scale_mpx
        wall_meas[i] = (computed_m, f"{computed_m:.2f} m")
        print(f"   Edge {i} ({edge_lens_px[i]:.0f}px) → {computed_m:.2f} m (computed)")

pts_m = best_pts * scale_mpx
pts_m[:, 0] -= pts_m[:, 0].min()
pts_m[:, 1] -= pts_m[:, 1].min()
span_x, span_y = pts_m[:, 0].max(), pts_m[:, 1].max()
wall_h = WALL_H_RATIO * (span_x + span_y) / 2.0
print(f"📦  {span_x:.2f}m × {span_y:.2f}m, wall_h={wall_h:.2f}m")

# Geometry
centroid = pts_m.mean(axis=0)
floor_pts = np.column_stack([pts_m[:, 0], pts_m[:, 1], np.zeros(n)])
ceiling_pts = np.column_stack([pts_m[:, 0], pts_m[:, 1], np.full(n, wall_h)])

wall_faces, wall_normals = [], []
for i in range(n):
    j = (i + 1) % n
    wall_faces.append([floor_pts[i], floor_pts[j], ceiling_pts[j], ceiling_pts[i]])
    dx, dy = pts_m[j, 0] - pts_m[i, 0], pts_m[j, 1] - pts_m[i, 1]
    nx, ny = dy, -dx
    mid = (pts_m[i] + pts_m[j]) / 2
    if nx * (mid[0] - centroid[0]) + ny * (mid[1] - centroid[1]) < 0: nx, ny = -nx, -ny
    wall_normals.append(np.array([nx, ny, 0]) / np.hypot(nx, ny))

elev_r, azim_r = np.radians(CAM_ELEV), np.radians(CAM_AZIM)
cam_dir = np.array([np.cos(elev_r)*np.cos(azim_r), np.cos(elev_r)*np.sin(azim_r), np.sin(elev_r)])
cam_right = np.array([np.sin(azim_r), -np.cos(azim_r), 0])

faces_cam = lambda n: float(np.dot(n, cam_dir)) > 0
shade = lambda n, f: ((TONE_R if np.dot(n[:2], cam_right[:2]) >= 0 else TONE_L),
                      (EDGE_R if np.dot(n[:2], cam_right[:2]) >= 0 else EDGE_L),
                      0.28 if f else 1.0)

back_walls = [(i, q) for i, q in enumerate(wall_faces) if not faces_cam(wall_normals[i])]
front_walls = [(i, q) for i, q in enumerate(wall_faces) if faces_cam(wall_normals[i])]

# Render
fig = plt.figure(figsize=(13, 9), facecolor=BG)
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG)
ax.grid(False)
ax.set_axis_off()
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill, pane.set_edgecolor = False, ("none",)

ax.add_collection3d(Poly3DCollection([floor_pts.tolist()], facecolor=FLOOR, edgecolor=FLOOR_E, alpha=1, linewidth=0.8))
grid = [[(x, 0, 0.001), (x, span_y, 0.001)] for x in np.arange(0, span_x+0.5, 0.5)]
grid += [[(0, y, 0.001), (span_x, y, 0.001)] for y in np.arange(0, span_y+0.5, 0.5)]
ax.add_collection3d(Line3DCollection(grid, colors=FLOOR_E, linewidths=0.4, alpha=0.5))

shadow = np.column_stack([pts_m[:, 0] + 0.04, pts_m[:, 1] + 0.04, np.full(n, -0.005)])
ax.add_collection3d(Poly3DCollection([shadow.tolist()], facecolor="#aaa", edgecolor="none", alpha=0.25))

for i, q in back_walls:
    fc, ec, a = shade(wall_normals[i], False)
    ax.add_collection3d(Poly3DCollection([q], facecolor=fc, edgecolor=ec, alpha=a, linewidth=0.7))
for i, q in front_walls:
    fc, ec, a = shade(wall_normals[i], True)
    ax.add_collection3d(Poly3DCollection([q], facecolor=fc, edgecolor=ec, alpha=a, linewidth=1, zorder=10))

# AO
for i, q in back_walls + front_walls:
    p0, p1 = np.array(q[0]), np.array(q[1])
    c3d = np.array([centroid[0], centroid[1], 0])
    for step in range(10):
        t0, t1 = step * 0.055, (step + 1) * 0.055
        fade = ((1 - step / 10) ** 2) * 0.07
        def ins(p, t):
            v = c3d - p; v[2] = 0; v /= (np.linalg.norm(v) + 1e-9)
            r = p + v * t; r[2] = 0.004; return r
        s0a, s1a = ins(p0.copy(), t0), ins(p1.copy(), t0)
        s0b, s1b = ins(p0.copy(), t1), ins(p1.copy(), t1)
        ax.add_collection3d(Poly3DCollection([[s0a, s1a, s1b, s0b]], facecolor="#111", edgecolor="none", alpha=fade, zorder=4))

ax.add_collection3d(Line3DCollection([[ceiling_pts[i], ceiling_pts[(i + 1) % n]] for i in range(n)], colors="#555566", linewidths=1.2, alpha=0.9, zorder=11))
ax.add_collection3d(Line3DCollection([[floor_pts[i], ceiling_pts[i]] for i in range(n)], colors="#555566", linewidths=1, alpha=0.85, zorder=11))

# Labels: ABOVE back, BELOW front
front_idx = {i for i, _ in front_walls}
for i, (_, lbl) in wall_meas.items():
    mid = (ceiling_pts[i] + ceiling_pts[(i + 1) % n]) / 2
    is_front = i in front_idx
    out = wall_normals[i] * (0.55 if is_front else 0.25)
    z_off = -0.35 if is_front else 0.15
    ax.text(mid[0] + out[0], mid[1] + out[1], mid[2] + z_off, lbl,
            fontsize=7.5, fontweight="bold", color="#222", ha="center",
            va=("top" if is_front else "bottom"),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#aaa", alpha=0.95, linewidth=0.6),
            zorder=50)

ax.set_xlim(-0.3, span_x + 0.3)
ax.set_ylim(-0.3, span_y + 0.3)
ax.set_zlim(-0.6, wall_h + 0.3)
ax.set_box_aspect([span_x, span_y, wall_h])
ax.view_init(elev=CAM_ELEV, azim=CAM_AZIM)

plt.tight_layout()
plt.savefig("isometric_output.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("💾  isometric_output.png")
plt.show()
