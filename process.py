"""
process.py - UNIVERSAL polygon-to-isometric converter
Handles: convex, concave, L-shaped, U-shaped, notched, any vertex count.
Color-coded fixtures: green=opening, pink=sink, magenta=fridge,
yellow=window, red=oven, blue=door.
"""

import io, re, math
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
LABEL_MAX_D  = 180

# ── fixture standard dimensions (metres) ──
FIXTURE_DEFS = {
    "sink":    {"depth": 0.55, "height": 0.85, "color": "#4FC3F7", "edge": "#0288D1", "label": "Sink"},
    "oven":    {"depth": 0.60, "height": 0.85, "color": "#EF5350", "edge": "#C62828", "label": "Oven"},
    "fridge":  {"depth": 0.60, "height": 1.80, "color": "#AB47BC", "edge": "#6A1B9A", "label": "Fridge"},
    "window":  {"depth": 0.05, "h_bot": 0.90, "h_top": 2.10, "color": "#81D4FA", "edge": "#0277BD", "label": "Window"},
    "door":    {"depth": 0.05, "height": 2.10, "color": "#A1887F", "edge": "#4E342E", "label": "Door"},
    "opening": None,
}

# ─────────────────────── geometry helpers ───────────────────────

def _edge_len(pts, i, j=None):
    if j is None: j = (i + 1) % len(pts)
    return np.hypot(pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1])

def _polygon_is_valid(pts):
    n = len(pts)
    if n < 3: return False
    perimeter = sum(_edge_len(pts, i) for i in range(n))
    if perimeter < 1e-6: return False
    min_edge = perimeter * 0.008
    return all(_edge_len(pts, i) >= min_edge for i in range(n))

def _remove_collinear(pts, angle_thresh_deg=6):
    out, n = [], len(pts)
    for i in range(n):
        p0, p1, p2 = pts[(i-1)%n], pts[i], pts[(i+1)%n]
        v1, v2 = p0 - p1, p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        angle = math.degrees(math.acos(np.clip(cos_a, -1, 1)))
        if abs(angle - 180) > angle_thresh_deg:
            out.append(p1)
    return np.array(out) if len(out) >= 3 else pts

def _detect_polygon(contour):
    perimeter = cv2.arcLength(contour, True)
    if cv2.contourArea(contour) < 1:
        return contour.reshape(-1, 2).astype(float)
    contour_pts = contour.reshape(-1, 2).astype(float)
    def max_deviation(approx_pts):
        n_a, max_d = len(approx_pts), 0
        step = max(1, len(contour_pts) // 200)
        for cp in contour_pts[::step]:
            min_d = float('inf')
            for i in range(n_a):
                j = (i + 1) % n_a
                min_d = min(min_d, _perp_dist(cp[0], cp[1], approx_pts[i], approx_pts[j]))
            max_d = max(max_d, min_d)
        return max_d
    candidates = []
    for eps_mult in np.linspace(0.002, 0.06, 50):
        approx = cv2.approxPolyDP(contour, eps_mult * perimeter, True)
        pts = _remove_collinear(approx.reshape(-1, 2).astype(float))
        if not _polygon_is_valid(pts): continue
        candidates.append((max_deviation(pts), len(pts), pts))
    if not candidates:
        return _remove_collinear(contour.reshape(-1, 2).astype(float), 5)
    candidates.sort(key=lambda x: x[1])
    min_dev = min(c[0] for c in candidates)
    acceptable = sorted([c for c in candidates if c[0] < min_dev * 2.0 + 5.0], key=lambda x: x[1])
    return acceptable[0][2]

def _signed_area(pts):
    n = len(pts)
    return sum(pts[i,0]*pts[(i+1)%n,1] - pts[(i+1)%n,0]*pts[i,1] for i in range(n)) / 2.0

def _ensure_ccw(pts):
    return pts[::-1].copy() if _signed_area(pts) < 0 else pts.copy()

def _outward_normal_2d(pts, i):
    j = (i + 1) % len(pts)
    dx, dy = pts[j,0]-pts[i,0], pts[j,1]-pts[i,1]
    nx, ny = dy, -dx
    length = math.hypot(nx, ny)
    return np.array([nx/length, ny/length]) if length > 1e-12 else np.array([0.0, 0.0])

def _perp_dist(px, py, p0, p1):
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    l2 = dx*dx + dy*dy
    if l2 < 1e-9: return math.hypot(px-p0[0], py-p0[1])
    t = max(0, min(1, ((px-p0[0])*dx + (py-p0[1])*dy) / l2))
    return math.hypot(px - (p0[0]+t*dx), py - (p0[1]+t*dy))

def _point_to_edge_t(px, py, p0, p1):
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    l2 = dx*dx + dy*dy
    if l2 < 1e-9: return 0.5
    return max(0.0, min(1.0, ((px-p0[0])*dx + (py-p0[1])*dy) / l2))

# ─────────────────────── OCR ───────────────────────

def _ocr_labels(img_bgr, w_img, h_img):
    try:
        scale = 3
        big = cv2.resize(img_bgr, (w_img*scale, h_img*scale), interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(big, config="--psm 11", output_type=pytesseract.Output.DICT)
        tokens = []
        for i, t in enumerate(data["text"]):
            t = t.strip()
            if t:
                cx = (data["left"][i] + data["width"][i]//2) // scale
                cy = (data["top"][i] + data["height"][i]//2) // scale
                tokens.append((t, cx, cy))
        labels, skip = [], set()
        for i, (t, cx, cy) in enumerate(tokens):
            if i in skip: continue
            m = re.match(r"(\d+[.,]\d+)\s*[mM]$", t)
            if m:
                labels.append((float(m.group(1).replace(",",".")), cx, cy)); continue
            if re.match(r"\d+[.,]\d+$", t) and i+1 < len(tokens):
                nt, nx, ny = tokens[i+1]
                if re.match(r"[mM]$", nt) and abs(nx-cx) < 100:
                    labels.append((float(t.replace(",",".")), (cx+nx)//2, (cy+ny)//2))
                    skip.add(i+1); continue
            if re.match(r"\d+[.,]\d+$", t):
                if i+1 < len(tokens) and re.match(r"[°]", tokens[i+1][0]): continue
                labels.append((float(t.replace(",",".")), cx, cy))
        return labels
    except Exception:
        return []

def _assign_labels(pts, labels):
    n = len(pts)
    assignments, used = {}, set()
    triples = []
    for li, (val, lx, ly) in enumerate(labels):
        for ei in range(n):
            d = _perp_dist(lx, ly, pts[ei], pts[(ei+1)%n])
            if d < LABEL_MAX_D: triples.append((d, ei, li))
    triples.sort()
    for d, ei, li in triples:
        if ei in assignments or li in used: continue
        val = labels[li][0]
        assignments[ei] = (val, f"{val:.2f} m")
        used.add(li)
    return assignments

# ─────────────────────── COLOR DETECTION ───────────────────────

def _detect_colored_segments(img_bgr, pts_px):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    n = len(pts_px)
    sat_mask = hsv[:,:,1] > 55
    val_mask = hsv[:,:,2] > 75
    color_mask = (sat_mask & val_mask).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(color_mask, 8)

    segments = []
    for comp_id in range(1, num_labels):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area < 80: continue
        comp_mask = (labels_map == comp_id)
        pb = img_bgr[comp_mask]
        ph = hsv[comp_mask]
        mean_b, mean_g, mean_r = pb.mean(axis=0).astype(float)
        mean_h, mean_s = ph[:,0].mean(), ph[:,1].mean()

        ft = _classify_color(mean_h, mean_s, mean_b, mean_g, mean_r)
        if ft is None: continue

        ys, xs = np.where(comp_mask)
        comp_points = np.column_stack([xs, ys]).astype(float)

        best_edge, best_dist = -1, float('inf')
        for ei in range(n):
            ej = (ei+1) % n
            step = max(1, len(comp_points)//50)
            med = np.median([_perp_dist(p[0],p[1],pts_px[ei],pts_px[ej]) for p in comp_points[::step]])
            if med < best_dist: best_dist, best_edge = med, ei
        if best_edge < 0 or best_dist > 30: continue

        ei, ej = best_edge, (best_edge+1) % n
        step = max(1, len(comp_points)//100)
        t_vals = [_point_to_edge_t(p[0],p[1],pts_px[ei],pts_px[ej]) for p in comp_points[::step]]
        segments.append({
            "edge_idx": ei, "t_start": min(t_vals), "t_end": max(t_vals),
            "fixture_type": ft, "width_px": (max(t_vals)-min(t_vals))*_edge_len(pts_px,ei),
        })
    return segments

def _classify_color(h, s, b, g, r):
    if g > r and g > b and 35 < h < 85 and g-r > 20: return "opening"
    if b > r and b > g and 85 < h < 130: return "door"
    if r > b and g > b and (g-b) > 30 and 10 < h < 40: return "window"
    if r > 170 and r-g > 60 and r-b > 60 and abs(g-b) < 40: return "oven"
    if r > g and r > b and b > g and h > 155 and (r-b) > 20: return "sink"
    if r > g and b > g and abs(r-b) < 30 and 130 < h < 165: return "fridge"
    if r > b and r > g and b > g and h > 150 and s > 50: return "sink"
    if r > g and b > g and abs(r-b) < 50 and (r+b)/2 - g > 30: return "fridge"
    if r > g and r > b and r - max(g,b) > 50: return "oven"
    return None

# ─────────────────────── depth-sorted face renderer ───────────────────────

def _render_all_faces(ax, faces, cam_dir):
    """
    Render all faces using a SINGLE Poly3DCollection with faces pre-sorted
    back-to-front. This prevents matplotlib from re-sorting them incorrectly.
    Each face: (verts_list, facecolor, edgecolor, alpha, linewidth)
    """
    # Compute depth for each face
    decorated = []
    for i, (verts, fc, ec, alpha, lw) in enumerate(faces):
        center = np.mean(verts, axis=0)
        depth = -np.dot(center, cam_dir)
        decorated.append((depth, i, verts, fc, ec, alpha, lw))
    # Sort back-to-front (farthest first)
    decorated.sort(key=lambda x: x[0], reverse=True)
    
    # Add each face as its own collection (one face per collection = no internal re-sort)
    # But set _sort_zpos to control matplotlib's inter-collection sorting
    for idx, (depth, _, verts, fc, ec, alpha, lw) in enumerate(decorated):
        pc = Poly3DCollection([verts], facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=lw)
        pc._sort_zpos = -depth  # matplotlib uses _sort_zpos for sorting collections
        ax.add_collection3d(pc)

# ─────────────────────── wall quad helper ───────────────────────

def _make_wall_quad(floor_3d, ceiling_3d, edge_i, t_start, t_end, z_bot, z_top):
    n = len(floor_3d)
    j = (edge_i + 1) % n
    p0 = floor_3d[edge_i] * (1-t_start) + floor_3d[j] * t_start
    p1 = floor_3d[edge_i] * (1-t_end)   + floor_3d[j] * t_end
    q0 = p0.copy(); q0[2] = z_bot
    q1 = p1.copy(); q1[2] = z_bot
    q2 = p1.copy(); q2[2] = z_top
    q3 = p0.copy(); q3[2] = z_top
    return [q0, q1, q2, q3]

# ─────────────────────── MAIN ───────────────────────

def generate_isometric(img_bgr: np.ndarray) -> bytes:
    h_img, w_img = img_bgr.shape[:2]

    # 1. contour
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("No contours found")
    cnt = max(contours, key=cv2.contourArea)

    # 2. polygon
    pts_px = _ensure_ccw(_detect_polygon(cnt))
    n = len(pts_px)

    # 3. OCR
    labels_ocr = _ocr_labels(img_bgr, w_img, h_img)
    wall_meas = _assign_labels(pts_px, labels_ocr)

    # 4. colored segments
    colored_segments = _detect_colored_segments(img_bgr, pts_px)
    print(f"[DEBUG] Found {len(colored_segments)} colored segments:")
    for seg in colored_segments:
        print(f"  edge {seg['edge_idx']}: {seg['fixture_type']} t=[{seg['t_start']:.3f},{seg['t_end']:.3f}]")

    # 5. scale
    edge_px = [_edge_len(pts_px, i) for i in range(n)]
    if wall_meas:
        scales = [v/edge_px[ei] for ei,(v,_) in wall_meas.items() if edge_px[ei] > 0]
        scale = float(np.median(scales)) if scales else 4.0/max(edge_px)
    else:
        scale = 4.0/max(edge_px)
    for i in range(n):
        if i not in wall_meas:
            m = edge_px[i] * scale
            wall_meas[i] = (m, f"{m:.2f} m")

    # 6. 3D points
    pts_m = pts_px * scale
    pts_m[:,0] -= pts_m[:,0].min()
    pts_m[:,1] -= pts_m[:,1].min()
    floor_3d   = np.column_stack([pts_m[:,0], pts_m[:,1], np.zeros(n)])
    ceiling_3d = np.column_stack([pts_m[:,0], pts_m[:,1], np.full(n, WALL_H)])

    # Build fixture lookup
    opening_segs = {}
    fixture_segs = {}
    for seg in colored_segments:
        ei = seg["edge_idx"]
        if seg["fixture_type"] == "opening":
            opening_segs.setdefault(ei, []).append((seg["t_start"], seg["t_end"]))
        else:
            fixture_segs.setdefault(ei, []).append((seg["t_start"], seg["t_end"], seg["fixture_type"]))

    # Camera
    elev_r, azim_r = np.radians(CAM_ELEV), np.radians(CAM_AZIM)
    cam_dir = np.array([np.cos(elev_r)*np.cos(azim_r), np.cos(elev_r)*np.sin(azim_r), np.sin(elev_r)])

    def lerp_edge(ei, t, arr):
        j = (ei + 1) % n
        return arr[ei] * (1 - t) + arr[j] * t

    def get_wall_gaps(ei):
        gaps = list(opening_segs.get(ei, []))
        for ts, te, ft in fixture_segs.get(ei, []):
            if ft == "door": gaps.append((ts, te))
        return gaps

    def get_solid_intervals(ei):
        gaps = get_wall_gaps(ei)
        if not gaps: return [(0.0, 1.0)]
        gaps.sort()
        merged = [gaps[0]]
        for gs, ge in gaps[1:]:
            if gs <= merged[-1][1]+0.001: merged[-1] = (merged[-1][0], max(merged[-1][1], ge))
            else: merged.append((gs, ge))
        solid, prev = [], 0.0
        for gs, ge in merged:
            if gs > prev+0.001: solid.append((prev, gs))
            prev = ge
        if prev < 0.999: solid.append((prev, 1.0))
        return solid

    # ── Collect ALL faces for depth-sorted rendering ──
    all_faces = []
    all_lines = []
    all_texts = []

    # Floor
    all_faces.append((floor_3d.tolist(), FLOOR_CLR, FLOOR_EDGE, 1.0, 0.8))

    # ── WALLS ──
    for i in range(n):
        j = (i + 1) % n
        normal_2d = _outward_normal_2d(pts_m, i)
        normal_3d = np.array([normal_2d[0], normal_2d[1], 0.0])
        dot = np.dot(normal_3d, cam_dir)
        is_back = dot <= 0
        clr = WALL_DARK if is_back else WALL_LIGHT
        alpha = 1.0 if is_back else 0.45

        solid = get_solid_intervals(i)
        fix_on_edge = [(ts,te,ft) for ts,te,ft in fixture_segs.get(i,[]) if ft != "door"]
        win_on_edge = [(ts,te) for ts,te,ft in fix_on_edge if ft == "window"]
        counter_on_edge = [(ts,te,ft) for ts,te,ft in fix_on_edge if ft in ("sink","oven","fridge")]

        for s_start, s_end in solid:
            sub_regions = []
            for ws, we in win_on_edge:
                if ws < s_end and we > s_start:
                    sub_regions.append((max(ws,s_start), min(we,s_end), "window"))
            for cs, ce, ct in counter_on_edge:
                if cs < s_end and ce > s_start:
                    sub_regions.append((max(cs,s_start), min(ce,s_end), ct))
            sub_regions.sort()

            if not sub_regions:
                all_faces.append((_make_wall_quad(floor_3d,ceiling_3d,i,s_start,s_end,0,WALL_H), clr, EDGE_CLR, alpha, 0.8))
            else:
                prev_t = s_start
                for rs, re, rtype in sub_regions:
                    if rs > prev_t + 0.001:
                        all_faces.append((_make_wall_quad(floor_3d,ceiling_3d,i,prev_t,rs,0,WALL_H), clr, EDGE_CLR, alpha, 0.8))
                    if rtype == "window":
                        fd = FIXTURE_DEFS["window"]
                        all_faces.append((_make_wall_quad(floor_3d,ceiling_3d,i,rs,re,0,fd["h_bot"]), clr, EDGE_CLR, alpha, 0.8))
                        all_faces.append((_make_wall_quad(floor_3d,ceiling_3d,i,rs,re,fd["h_top"],WALL_H), clr, EDGE_CLR, alpha, 0.8))
                    else:
                        fh = FIXTURE_DEFS.get(rtype,{}).get("height",0.85)
                        all_faces.append((_make_wall_quad(floor_3d,ceiling_3d,i,rs,re,fh,WALL_H), clr, EDGE_CLR, alpha, 0.8))
                    prev_t = re
                if prev_t < s_end - 0.001:
                    all_faces.append((_make_wall_quad(floor_3d,ceiling_3d,i,prev_t,s_end,0,WALL_H), clr, EDGE_CLR, alpha, 0.8))

        # Top edges
        for s_start, s_end in solid:
            c0 = lerp_edge(i, s_start, ceiling_3d)
            c1 = lerp_edge(i, s_end, ceiling_3d)
            all_lines.append((c0,c1,EDGE_CLR,1.2,0.9))
        # Vertical edges
        gaps = get_wall_gaps(i)
        if not any(gs < 0.01 for gs,ge in gaps):
            all_lines.append((floor_3d[i], ceiling_3d[i], EDGE_CLR, 1.0, 0.85))
        for gs, ge in gaps:
            if gs > 0.01:
                all_lines.append((lerp_edge(i,gs,floor_3d), lerp_edge(i,gs,ceiling_3d), EDGE_CLR, 1.0, 0.85))
            if ge < 0.99:
                all_lines.append((lerp_edge(i,ge,floor_3d), lerp_edge(i,ge,ceiling_3d), EDGE_CLR, 1.0, 0.85))

    # ── FIXTURES ──
    for seg in colored_segments:
        ft = seg["fixture_type"]
        if ft == "opening": continue
        ei = seg["edge_idx"]
        ts, te = seg["t_start"], seg["t_end"]
        normal_2d = _outward_normal_2d(pts_m, ei)
        normal_3d = np.array([normal_2d[0], normal_2d[1], 0.0])
        inward_3d = -normal_3d
        p0_3d = lerp_edge(ei, ts, floor_3d)
        p1_3d = lerp_edge(ei, te, floor_3d)
        seg_w = np.linalg.norm(p1_3d - p0_3d)
        fdef = FIXTURE_DEFS[ft]

        if ft == "window":
            h_bot, h_top = fdef["h_bot"], fdef["h_top"]
            off = inward_3d * 0.02
            w0 = p0_3d+off; w0 = w0.copy(); w0[2] = h_bot
            w1 = p1_3d+off; w1 = w1.copy(); w1[2] = h_bot
            w2 = p1_3d+off; w2 = w2.copy(); w2[2] = h_top
            w3 = p0_3d+off; w3 = w3.copy(); w3[2] = h_top
            all_faces.append(([w0,w1,w2,w3], "#B3E5FC", "#0277BD", 0.5, 1.2))
            for a,b in [(w0,w1),(w1,w2),(w2,w3),(w3,w0)]:
                all_lines.append((a,b,"#0277BD",1.5,0.9))
            mid_h = (h_bot+h_top)/2
            mc = (p0_3d+p1_3d)/2+off
            m_bot = mc.copy(); m_bot[2] = h_bot
            m_top = mc.copy(); m_top[2] = h_top
            all_lines.append((m_bot, m_top, "#0277BD", 1.0, 0.7))
            mh0 = (p0_3d+off).copy(); mh0[2] = mid_h
            mh1 = (p1_3d+off).copy(); mh1[2] = mid_h
            all_lines.append((mh0, mh1, "#0277BD", 1.0, 0.7))
            lp = (w2+w3)/2 + normal_3d*0.15; lp[2] += 0.1
            all_texts.append((lp[0],lp[1],lp[2],"Window",
                dict(fontsize=6.5,fontweight="bold",color="#01579B",ha="center",va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2",facecolor="#E1F5FE",edgecolor="#0277BD",alpha=0.9,linewidth=0.5))))
            continue

        if ft == "door":
            h = fdef["height"]
            off = inward_3d * 0.03
            d0 = (p0_3d+off).copy(); d0[2] = 0
            d1 = (p1_3d+off).copy(); d1[2] = 0
            d2 = (p1_3d+off).copy(); d2[2] = h
            d3 = (p0_3d+off).copy(); d3[2] = h
            all_faces.append(([d0,d1,d2,d3], "#8D6E63", "#4E342E", 0.55, 1.0))
            for a,b in [(d0,d1),(d1,d2),(d2,d3),(d3,d0)]:
                all_lines.append((a,b,"#4E342E",1.5,0.9))
            # Handle
            handle = d1*0.85 + d0*0.15; handle[2] = 1.0
            all_lines.append((handle, handle + inward_3d*0.001, "#FFD54F", 3.0, 1.0))
            lp = (d2+d3)/2 + normal_3d*0.15; lp[2] = h+0.1
            all_texts.append((lp[0],lp[1],lp[2],"Door",
                dict(fontsize=6.5,fontweight="bold",color="#3E2723",ha="center",va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2",facecolor="#EFEBE9",edgecolor="#5D4037",alpha=0.9,linewidth=0.5))))
            continue

        # Counter fixtures: sink, oven, fridge
        depth = fdef["depth"]
        height = fdef["height"]
        color = fdef["color"]
        ec = fdef["edge"]
        label = fdef["label"]

        b0 = p0_3d.copy(); b0[2] = 0
        b1 = p1_3d.copy(); b1[2] = 0
        b2 = (p1_3d + inward_3d*depth).copy(); b2[2] = 0
        b3 = (p0_3d + inward_3d*depth).copy(); b3[2] = 0
        t0 = b0.copy(); t0[2] = height
        t1 = b1.copy(); t1[2] = height
        t2 = b2.copy(); t2[2] = height
        t3 = b3.copy(); t3[2] = height

        all_faces.append(([t0.copy(),t1.copy(),t2.copy(),t3.copy()], color, ec, 0.92, 0.9))
        all_faces.append(([b2.copy(),b3.copy(),t3.copy(),t2.copy()], color, ec, 0.90, 0.9))
        all_faces.append(([b0.copy(),b3.copy(),t3.copy(),t0.copy()], color, ec, 0.82, 0.7))
        all_faces.append(([b1.copy(),b2.copy(),t2.copy(),t1.copy()], color, ec, 0.82, 0.7))
        all_faces.append(([b0.copy(),b1.copy(),t1.copy(),t0.copy()], color, ec, 0.40, 0.4))

        for a,b in [(t0,t1),(t1,t2),(t2,t3),(t3,t0)]:
            all_lines.append((a.copy(),b.copy(),ec,1.0,0.9))
        for a,b in [(b0,t0),(b1,t1),(b2,t2),(b3,t3)]:
            all_lines.append((a.copy(),b.copy(),ec,0.7,0.8))

        edge_dir_n = (p1_3d - p0_3d) / (np.linalg.norm(p1_3d - p0_3d) + 1e-9)

        if ft == "sink":
            s0 = (p0_3d + edge_dir_n*seg_w*0.1 + inward_3d*depth*0.15).copy(); s0[2] = height-0.02
            s1 = (p1_3d - edge_dir_n*seg_w*0.1 + inward_3d*depth*0.15).copy(); s1[2] = height-0.02
            s2 = (p1_3d - edge_dir_n*seg_w*0.1 + inward_3d*depth*0.85).copy(); s2[2] = height-0.02
            s3 = (p0_3d + edge_dir_n*seg_w*0.1 + inward_3d*depth*0.85).copy(); s3[2] = height-0.02
            all_faces.append(([s0,s1,s2,s3], "#E1F5FE", "#0288D1", 0.85, 0.8))
        elif ft == "oven":
            od0 = b3.copy() + edge_dir_n*seg_w*0.1; od0[2] = 0.12
            od1 = b2.copy() - edge_dir_n*seg_w*0.1; od1[2] = 0.12
            od2 = b2.copy() - edge_dir_n*seg_w*0.1; od2[2] = height*0.55
            od3 = b3.copy() + edge_dir_n*seg_w*0.1; od3[2] = height*0.55
            all_faces.append(([od0,od1,od2,od3], "#FFCDD2", "#C62828", 0.75, 0.8))
        elif ft == "fridge":
            split_h = height * 0.6
            fl0 = b3.copy() + edge_dir_n*seg_w*0.05; fl0[2] = split_h
            fl1 = b2.copy() - edge_dir_n*seg_w*0.05; fl1[2] = split_h
            all_lines.append((fl0, fl1, "#4A148C", 1.2, 0.8))

        lp = (t0+t1+t2+t3)/4; lp[2] += 0.12
        all_texts.append((lp[0],lp[1],lp[2], label,
            dict(fontsize=6.5,fontweight="bold",color=ec,ha="center",va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2",facecolor="white",edgecolor=ec,alpha=0.9,linewidth=0.5))))

    # ── Setup figure ──
    span_x = pts_m[:,0].max() - pts_m[:,0].min()
    span_y = pts_m[:,1].max() - pts_m[:,1].min()
    fig = plt.figure(figsize=(13, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    ax.grid(False); ax.set_axis_off()
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor("none")

    # Render depth-sorted faces
    _render_all_faces(ax, all_faces, cam_dir)

    # Floor grid
    path_2d = MplPath(pts_m)
    for x in np.arange(0, span_x+0.5, 0.5):
        segs = [(x,y) for y in np.arange(0, span_y+0.5, 0.05) if path_2d.contains_point((x,y))]
        if len(segs) >= 2:
            ax.plot([segs[0][0],segs[-1][0]], [segs[0][1],segs[-1][1]], [0.001,0.001],
                    color=FLOOR_EDGE, linewidth=0.35, alpha=0.45)
    for y in np.arange(0, span_y+0.5, 0.5):
        segs = [(x,y) for x in np.arange(0, span_x+0.5, 0.05) if path_2d.contains_point((x,y))]
        if len(segs) >= 2:
            ax.plot([segs[0][0],segs[-1][0]], [segs[0][1],segs[-1][1]], [0.001,0.001],
                    color=FLOOR_EDGE, linewidth=0.35, alpha=0.45)

    # Lines
    for p0, p1, clr, lw, alpha in all_lines:
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]], color=clr, linewidth=lw, alpha=alpha)

    # Measurement labels
    for i, (_, lbl) in wall_meas.items():
        j = (i + 1) % n
        mid_top = (ceiling_3d[i] + ceiling_3d[j]) / 2.0
        normal_2d = _outward_normal_2d(pts_m, i)
        offset = np.array([normal_2d[0], normal_2d[1], 0.0]) * 0.25
        ax.text(mid_top[0]+offset[0], mid_top[1]+offset[1], mid_top[2]+0.15,
                lbl, fontsize=7.5, fontweight="bold", color="#222", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#aaa", alpha=0.95, linewidth=0.6),
                zorder=50)

    # Fixture labels
    for x, y, z, text, kwargs in all_texts:
        ax.text(x, y, z, text, zorder=50, **kwargs)

    # Axes
    pad = 0.5
    ax.set_xlim(pts_m[:,0].min()-pad, pts_m[:,0].max()+pad)
    ax.set_ylim(pts_m[:,1].min()-pad, pts_m[:,1].max()+pad)
    ax.set_zlim(-0.3, WALL_H+pad)
    ax.set_box_aspect([max(span_x,0.1), max(span_y,0.1), WALL_H])
    ax.view_init(elev=CAM_ELEV, azim=CAM_AZIM)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_input.png"
    img = cv2.imread(path)
    if img is None: print(f"Cannot read {path}"); sys.exit(1)
    result = generate_isometric(img)
    out_path = path.rsplit(".", 1)[0] + "_isometric.png"
    with open(out_path, "wb") as f: f.write(result)
    print(f"Saved: {out_path}")
