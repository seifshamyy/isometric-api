"""
process.py - Core isometric rendering with OCR-based measurement matching
"""

import io, re
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pytesseract

WALL_H_RATIO, APPROX_EPS = 0.5, 0.02
CAM_ELEV, CAM_AZIM = 48, -30
BG, FLOOR, FLOOR_E = "#dde0e5", "#c8c8c8", "#c8c8c8"
TONE_R, TONE_L, EDGE_R, EDGE_L = "#909090", "#7a7a7a", "#707070", "#2a2a2a"
MAX_LABEL_DIST = 150

def generate_isometric(img_bgr: np.ndarray) -> bytes:
    """Takes BGR image, returns PNG bytes of isometric render"""
    h_img, w_img = img_bgr.shape[:2]

    # Detect polygon
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("No contours found")

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

    # OCR
    try:
        big = cv2.resize(img_bgr, (w_img*3, h_img*3), interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(big, config="--psm 11", output_type=pytesseract.Output.DICT)
        tokens = [(t.strip(), (data["left"][i] + data["width"][i]//2)//3, (data["top"][i] + data["height"][i]//2)//3)
                  for i, t in enumerate(data["text"]) if t.strip()]

        labels, skip = [], set()
        for i, (t, cx, cy) in enumerate(tokens):
            if i in skip: continue
            m = re.match(r"(\d+[.,]\d+)\s*[mM]$", t)
            if m:
                labels.append((float(m.group(1).replace(",", ".")), cx, cy))
            elif re.match(r"\d+[.,]\d+$", t) and i+1 < len(tokens):
                nt, nx, ny = tokens[i+1]
                if re.match(r"[mM]$", nt) and abs(nx - cx) < 80:
                    labels.append((float(t.replace(",", ".")), (cx+nx)//2, (cy+ny)//2))
                    skip.add(i+1)
            elif re.match(r"(\d+)[mM]$", t) and i > 0:
                prev_t, prev_cx, prev_cy = tokens[i-1]
                if re.match(r"\d+\.\d*$", prev_t) and abs(prev_cx - cx) < 100:
                    combined = prev_t + t[:-1]
                    if re.match(r"\d+\.\d+$", combined):
                        labels.append((float(combined), (prev_cx + cx)//2, (prev_cy + cy)//2))
                        skip.add(i-1)
    except Exception as e:
        labels = []

    # Match labels to edges using perpendicular distance
    def perp_dist(px, py, p0, p1):
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0: return np.hypot(px - x0, py - y0)
        t = max(0, min(1, ((px - x0) * dx + (py - y0) * dy) / (dx*dx + dy*dy)))
        proj_x, proj_y = x0 + t * dx, y0 + t * dy
        return np.hypot(px - proj_x, py - proj_y)

    # Calculate edge lengths for OCR validation
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
            
            wall_meas[i] = (meas_m, f"{meas_m:.2f} m")
            used_labels.add(best_label_idx)

    # Scale
    if wall_meas:
        scales = []
        for i, (meas_m, _) in wall_meas.items():
            j = (i + 1) % n
            px_len = np.hypot(best_pts[j, 0] - best_pts[i, 0], best_pts[j, 1] - best_pts[i, 1])
            scales.append(meas_m / px_len)
        scale_mpx = np.median(scales)
    else:
        scale_mpx = 4.0 / max(edge_lens_px)

    # Add computed measurements for edges without OCR labels
    for i in range(n):
        if i not in wall_meas:
            computed_m = edge_lens_px[i] * scale_mpx
            wall_meas[i] = (computed_m, f"{computed_m:.2f} m")

    pts_m = best_pts * scale_mpx
    pts_m[:, 0] -= pts_m[:, 0].min()
    pts_m[:, 1] -= pts_m[:, 1].min()
    span_x, span_y = pts_m[:, 0].max(), pts_m[:, 1].max()
    wall_h = WALL_H_RATIO * (span_x + span_y) / 2.0

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
