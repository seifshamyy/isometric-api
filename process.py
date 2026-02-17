"""
process.py - UNIVERSAL polygon-to-isometric converter
Unified rendering: walls + fixtures are one integrated geometry pass.
Painter's algorithm: all faces sorted by camera depth, painted back-to-front.
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
WALL_H    = 2.8
CAM_ELEV  = 48
CAM_AZIM  = -45
BG        = "#dde0e5"
FLOOR_CLR = "#c8c8c8"
FLOOR_EDGE= "#b8b8b8"
WALL_LT   = "#909090"
WALL_DK   = "#7a7a7a"
EDGE_CLR  = "#555566"
LABEL_MAX = 180

# ── fixture specs ──
FIX = {
    "sink":   {"h": 0.80, "depth": 0.55, "clr": "#FF69B4", "ec": "#CC5590"},
    "fridge": {"h": 1.80, "depth": 0.65, "clr": "#CC00CC", "ec": "#990099"},
    "stove":  {"h": 0.85, "depth": 0.55, "clr": "#E03030", "ec": "#AA2020"},
    "window": {"bot": 0.90, "top": 2.10, "clr": "#FFE44D", "ec": "#CCA800"},
    "door":   {"h": 2.10, "min_w": 0.70, "clr": "#D2B48C", "ec": "#8a6d4a"},
}

# ── color detection ──
COLOR_RANGES = {
    "green":   [((35,60,60),(85,255,255))],
    "pink":    [((160,60,100),(175,255,255))],
    "magenta": [((145,60,50),(159,255,255))],
    "yellow":  [((15,60,80),(35,255,255))],
    "red":     [((0,100,80),(10,255,255)),((170,100,80),(180,255,255))],
    "blue":    [((90,60,80),(135,255,255))],
}
COLOR_MAP = {"green":"opening","pink":"sink","magenta":"fridge",
             "yellow":"window","red":"stove","blue":"door"}

# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS (original, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _edge_len(pts, i, j=None):
    if j is None: j = (i+1) % len(pts)
    return np.hypot(pts[j,0]-pts[i,0], pts[j,1]-pts[i,1])

def _polygon_is_valid(pts):
    n = len(pts)
    if n < 3: return False
    peri = sum(_edge_len(pts,i) for i in range(n))
    if peri < 1e-6: return False
    return all(_edge_len(pts,i) >= peri*0.008 for i in range(n))

def _remove_collinear(pts, thresh=6):
    out = []
    n = len(pts)
    for i in range(n):
        v1, v2 = pts[(i-1)%n]-pts[i], pts[(i+1)%n]-pts[i]
        cos_a = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12)
        if abs(math.degrees(math.acos(np.clip(cos_a,-1,1)))-180) > thresh:
            out.append(pts[i])
    return np.array(out) if len(out) >= 3 else pts

def _perp_dist(px, py, p0, p1):
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    l2 = dx*dx+dy*dy
    if l2 < 1e-9: return math.hypot(px-p0[0],py-p0[1])
    t = max(0,min(1,((px-p0[0])*dx+(py-p0[1])*dy)/l2))
    return math.hypot(px-(p0[0]+t*dx),py-(p0[1]+t*dy))

def _detect_polygon(contour):
    peri = cv2.arcLength(contour, True)
    if cv2.contourArea(contour) < 1:
        return contour.reshape(-1,2).astype(float)
    cpts = contour.reshape(-1,2).astype(float)
    def maxdev(ap):
        na = len(ap); step = max(1,len(cpts)//200)
        return max(min(_perp_dist(cp[0],cp[1],ap[i],ap[(i+1)%na]) for i in range(na)) for cp in cpts[::step])
    cands = []
    for em in np.linspace(0.002,0.06,50):
        pts = _remove_collinear(cv2.approxPolyDP(contour,em*peri,True).reshape(-1,2).astype(float))
        if _polygon_is_valid(pts): cands.append((maxdev(pts),len(pts),pts))
    if not cands:
        return _remove_collinear(cpts, 5)
    cands.sort(key=lambda x:x[1])
    md = min(c[0] for c in cands)
    return sorted([c for c in cands if c[0]<md*2+5],key=lambda x:x[1])[0][2]

def _signed_area(p):
    n=len(p); return sum(p[i,0]*p[(i+1)%n,1]-p[(i+1)%n,0]*p[i,1] for i in range(n))/2

def _ensure_ccw(p):
    return p[::-1].copy() if _signed_area(p)<0 else p.copy()

def _outward_normal(pts, i):
    j=(i+1)%len(pts); dx,dy=pts[j,0]-pts[i,0],pts[j,1]-pts[i,1]
    l=math.hypot(dy,-dx)
    return np.array([dy/l,-dx/l]) if l>1e-12 else np.array([0.,0.])

def _inward_normal(pts, i):
    return -_outward_normal(pts, i)

# ── OCR ──
def _ocr_labels(img, w, h):
    try:
        sc=3; big=cv2.resize(img,(w*sc,h*sc),interpolation=cv2.INTER_CUBIC)
        d=pytesseract.image_to_data(big,config="--psm 11",output_type=pytesseract.Output.DICT)
        toks=[(d["text"][i].strip(),(d["left"][i]+d["width"][i]//2)//sc,(d["top"][i]+d["height"][i]//2)//sc)
              for i in range(len(d["text"])) if d["text"][i].strip()]
        labels,skip=[],set()
        for i,(t,cx,cy) in enumerate(toks):
            if i in skip: continue
            m=re.match(r"(\d+[.,]\d+)\s*[mM]$",t)
            if m: labels.append((float(m.group(1).replace(",",".")) ,cx,cy)); continue
            if re.match(r"\d+[.,]\d+$",t) and i+1<len(toks):
                nt,nx,ny=toks[i+1]
                if re.match(r"[mM]$",nt) and abs(nx-cx)<100:
                    labels.append((float(t.replace(",",".")),(cx+nx)//2,(cy+ny)//2)); skip.add(i+1); continue
            if re.match(r"\d+[.,]\d+$",t):
                if i+1<len(toks) and re.match(r"[°]",toks[i+1][0]): continue
                labels.append((float(t.replace(",",".")),cx,cy))
        return labels
    except: return []

def _assign_labels(pts, labels):
    n=len(pts); asgn={}; used=set(); trips=[]
    for li,(v,lx,ly) in enumerate(labels):
        for ei in range(n):
            d=_perp_dist(lx,ly,pts[ei],pts[(ei+1)%n])
            if d<LABEL_MAX: trips.append((d,ei,li))
    trips.sort()
    for d,ei,li in trips:
        if ei in asgn or li in used: continue
        asgn[ei]=(labels[li][0],f"{labels[li][0]:.2f} m"); used.add(li)
    return asgn

# ── Color detection ──
def _detect_colors(img, pts):
    h,w=img.shape[:2]; hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); n=len(pts); out=[]
    kern=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    for cn,ranges in COLOR_RANGES.items():
        mask=np.zeros((h,w),np.uint8)
        for lo,hi in ranges: mask|=cv2.inRange(hsv,np.array(lo,np.uint8),np.array(hi,np.uint8))
        mask=cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kern,iterations=2),cv2.MORPH_OPEN,kern,iterations=1)
        cxy=np.argwhere(mask>0)[:,::-1].astype(float)
        if len(cxy)<10: continue
        for ei in range(n):
            ej=(ei+1)%n; ev=pts[ej]-pts[ei]; el=np.linalg.norm(ev)
            if el<5: continue
            ed=ev/el; vecs=cxy-pts[ei]; tv=vecs@ed
            perp=np.abs(vecs[:,0]*(-ed[1])+vecs[:,1]*ed[0])
            ok=(tv>=-5)&(tv<=el+5)&(perp<20); tvv=tv[ok]
            if len(tvv)<5: continue
            ts=max(0,np.percentile(tvv,2))/el; te=min(el,np.percentile(tvv,98))/el
            if te-ts<0.02: continue
            out.append({"edge_idx":ei,"fixture_type":COLOR_MAP[cn],"t_start":ts,"t_end":te,"color_name":cn})
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED FACE-BASED RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def _pt(floor, ceil, ei, ej, t, z):
    """Interpolate a 3D point along edge ei→ej at parametric t, with explicit z."""
    p = floor[ei] + t * (floor[ej] - floor[ei])
    p = p.copy(); p[2] = z
    return p

def _quad(a,b,c,d):
    return [np.array(x,dtype=float) for x in [a,b,c,d]]

def _face_depth(verts, cam_dir):
    c = np.mean([np.array(v) for v in verts], axis=0)
    return -np.dot(c, cam_dir)

def _build_scene(n, floor, ceil, pts_m, fixtures, cam_dir):
    """
    Build every renderable face + line + label for the entire scene.
    Returns: (faces, lines, labels)
    where faces = [(verts, fc, ec, alpha, lw), ...]
          lines = [(p0, p1, color, lw, alpha), ...]
          labels = [(x,y,z, text, fontsize, bbox_fc, bbox_ec), ...]
    """
    faces = []   # (verts_list, facecolor, edgecolor, alpha, linewidth)
    lines = []
    labels = []

    def add_face(v, fc, ec, a=1.0, lw=0.7):
        faces.append((v, fc, ec, a, lw))

    # ── Floor ──
    add_face(floor.tolist(), FLOOR_CLR, FLOOR_EDGE, 1.0, 0.8)

    # ── Per-edge: walls + fixtures ──
    for i in range(n):
        j = (i+1) % n
        out2 = _outward_normal(pts_m, i)
        out3 = np.array([out2[0], out2[1], 0.0])
        in3 = -out3
        dot = np.dot(out3, cam_dir)
        is_front = dot > 0
        wc = WALL_LT if is_front else WALL_DK
        wa = 0.45 if is_front else 1.0

        efx = sorted([f for f in fixtures if f["edge_idx"]==i], key=lambda f:f["t_start"])
        edge_len_m = np.linalg.norm(floor[j][:2] - floor[i][:2])

        cursor = 0.0
        for fx in efx:
            ts, te = fx["t_start"], fx["t_end"]
            ft = fx["fixture_type"]

            # ── solid wall before fixture ──
            if cursor < ts - 0.001:
                add_face(_quad(_pt(floor,ceil,i,j,cursor,0), _pt(floor,ceil,i,j,ts,0),
                               _pt(floor,ceil,i,j,ts,WALL_H), _pt(floor,ceil,i,j,cursor,WALL_H)),
                         wc, EDGE_CLR, wa)

            # ── fixture zone ──
            if ft == "opening":
                pass  # total gap

            elif ft == "door":
                fi = FIX["door"]; dh = fi["h"]
                # enforce min width
                mid = (ts+te)/2; seg_m = (te-ts)*edge_len_m
                if seg_m < fi["min_w"] and edge_len_m > 0:
                    ht = (fi["min_w"]/edge_len_m)/2
                    dts, dte = max(0,mid-ht), min(1,mid+ht)
                else:
                    dts, dte = ts, te

                # wall above door
                add_face(_quad(_pt(floor,ceil,i,j,dts,dh), _pt(floor,ceil,i,j,dte,dh),
                               _pt(floor,ceil,i,j,dte,WALL_H), _pt(floor,ceil,i,j,dts,WALL_H)),
                         wc, EDGE_CLR, wa)
                # door panel
                off = in3 * 0.04
                add_face(_quad(_pt(floor,ceil,i,j,dts,0)+off, _pt(floor,ceil,i,j,dte,0)+off,
                               _pt(floor,ceil,i,j,dte,dh)+off, _pt(floor,ceil,i,j,dts,dh)+off),
                         fi["clr"], fi["ec"], 0.6 if is_front else 0.8, 1.2)
                # frame lines
                for a,b in [(_pt(floor,ceil,i,j,dts,0)+off, _pt(floor,ceil,i,j,dts,dh)+off),
                            (_pt(floor,ceil,i,j,dte,0)+off, _pt(floor,ceil,i,j,dte,dh)+off),
                            (_pt(floor,ceil,i,j,dts,dh)+off, _pt(floor,ceil,i,j,dte,dh)+off)]:
                    lines.append((a,b,fi["ec"],1.5,0.85))
                # swing arc
                p0_2d = _pt(floor,ceil,i,j,dts,0)[:2]
                p1_2d = _pt(floor,ceil,i,j,dte,0)[:2]
                dw = np.linalg.norm(p1_2d-p0_2d)
                dd = (p1_2d-p0_2d)/(dw+1e-12); sw = in3[:2]
                arc = [np.append(p0_2d + dw*(math.cos(k/20*math.pi/2)*dd + math.sin(k/20*math.pi/2)*sw), 0.005) for k in range(21)]
                for k in range(20): lines.append((arc[k],arc[k+1],fi["ec"],0.8,0.4))
                # label
                lp = (_pt(floor,ceil,i,j,dts,dh)+_pt(floor,ceil,i,j,dte,dh))/2
                labels.append((lp[0],lp[1],lp[2]+0.12,"Door",6.5,fi["clr"],fi["ec"]))

            elif ft == "window":
                fi = FIX["window"]; wb,wt = fi["bot"],fi["top"]
                # wall below sill
                add_face(_quad(_pt(floor,ceil,i,j,ts,0), _pt(floor,ceil,i,j,te,0),
                               _pt(floor,ceil,i,j,te,wb), _pt(floor,ceil,i,j,ts,wb)),
                         wc, EDGE_CLR, wa)
                # wall above lintel
                add_face(_quad(_pt(floor,ceil,i,j,ts,wt), _pt(floor,ceil,i,j,te,wt),
                               _pt(floor,ceil,i,j,te,WALL_H), _pt(floor,ceil,i,j,ts,WALL_H)),
                         wc, EDGE_CLR, wa)
                # glass (slightly outward)
                off = out3 * 0.015
                add_face(_quad(_pt(floor,ceil,i,j,ts,wb)+off, _pt(floor,ceil,i,j,te,wb)+off,
                               _pt(floor,ceil,i,j,te,wt)+off, _pt(floor,ceil,i,j,ts,wt)+off),
                         fi["clr"], fi["ec"], 0.55 if is_front else 0.7, 1.0)
                # crossbars + frame
                mh=(wb+wt)/2; mt=(ts+te)/2
                lines.append((_pt(floor,ceil,i,j,ts,mh)+off, _pt(floor,ceil,i,j,te,mh)+off, fi["ec"],1.0,0.7))
                lines.append((_pt(floor,ceil,i,j,mt,wb)+off, _pt(floor,ceil,i,j,mt,wt)+off, fi["ec"],1.0,0.7))
                for a,b in [(_pt(floor,ceil,i,j,ts,wb)+off,_pt(floor,ceil,i,j,te,wb)+off),
                            (_pt(floor,ceil,i,j,te,wb)+off,_pt(floor,ceil,i,j,te,wt)+off),
                            (_pt(floor,ceil,i,j,te,wt)+off,_pt(floor,ceil,i,j,ts,wt)+off),
                            (_pt(floor,ceil,i,j,ts,wt)+off,_pt(floor,ceil,i,j,ts,wb)+off)]:
                    lines.append((a,b,fi["ec"],1.5,0.9))
                lp=(_pt(floor,ceil,i,j,ts,wt)+_pt(floor,ceil,i,j,te,wt))/2
                labels.append((lp[0],lp[1],lp[2]+0.12,"Window",6.5,"#FFF8B0",fi["ec"]))

            elif ft in ("sink","stove","fridge"):
                fi = FIX[ft]; fh=fi["h"]; fd=fi["depth"]

                if is_front:
                    # Front wall: full wall behind, fixture paints on top
                    add_face(_quad(_pt(floor,ceil,i,j,ts,0), _pt(floor,ceil,i,j,te,0),
                                   _pt(floor,ceil,i,j,te,WALL_H), _pt(floor,ceil,i,j,ts,WALL_H)),
                             wc, EDGE_CLR, wa)
                else:
                    # Back wall: only wall ABOVE fixture height (shorter quad won't occlude fixture)
                    if fh < WALL_H - 0.05:
                        add_face(_quad(_pt(floor,ceil,i,j,ts,fh), _pt(floor,ceil,i,j,te,fh),
                                       _pt(floor,ceil,i,j,te,WALL_H), _pt(floor,ceil,i,j,ts,WALL_H)),
                                 wc, wc, wa, 0.3)  # use wc as edge color to hide seam

                # Box extrudes inward (into the room)
                ps=_pt(floor,ceil,i,j,ts,0); pe=_pt(floor,ceil,i,j,te,0)
                ps[2]=0; pe[2]=0
                depth_vec = in3 * fd
                c0=ps.copy(); c1=pe.copy()
                c2=pe+depth_vec; c3=ps+depth_vec
                c4=c0.copy();c4[2]=fh; c5=c1.copy();c5[2]=fh
                c6=c2.copy();c6[2]=fh; c7=c3.copy();c7[2]=fh
                add_face(_quad(c0,c1,c5,c4), fi["clr"],fi["ec"],0.92,0.9)
                add_face(_quad(c3,c2,c6,c7), fi["clr"],fi["ec"],0.92,0.9)
                add_face(_quad(c0,c3,c7,c4), fi["clr"],fi["ec"],0.92,0.9)
                add_face(_quad(c1,c2,c6,c5), fi["clr"],fi["ec"],0.92,0.9)
                add_face(_quad(c4,c5,c6,c7), fi["clr"],fi["ec"],0.92,0.9)
                add_face(_quad(c0,c1,c2,c3), fi["clr"],fi["ec"],0.92,0.9)
                ct=(c4+c5+c6+c7)/4
                labels.append((ct[0],ct[1],ct[2]+0.08,ft.capitalize(),6.5,"white","#999"))

            cursor = te

        # solid wall after last fixture
        if cursor < 0.999:
            add_face(_quad(_pt(floor,ceil,i,j,cursor,0), _pt(floor,ceil,i,j,1,0),
                           _pt(floor,ceil,i,j,1,WALL_H), _pt(floor,ceil,i,j,cursor,WALL_H)),
                     wc, EDGE_CLR, wa)

        # ── structural lines for this edge ──
        openings = [(f["t_start"],f["t_end"]) for f in efx if f["fixture_type"]=="opening"]
        for z_line, lw_line, a_line in [(WALL_H, 1.2, 0.9), (0.001, 0.7, 0.5)]:
            cur = 0.0
            for os,oe in sorted(openings):
                if cur < os-0.001:
                    lines.append((_pt(floor,ceil,i,j,cur,z_line),_pt(floor,ceil,i,j,os,z_line),EDGE_CLR,lw_line,a_line))
                cur = oe
            if cur < 0.999:
                lines.append((_pt(floor,ceil,i,j,cur,z_line),_pt(floor,ceil,i,j,1,z_line),EDGE_CLR,lw_line,a_line))

    # ── vertical corner edges ──
    for i in range(n):
        prev = (i-1) % n
        skip = False
        for f in fixtures:
            if f["fixture_type"]!="opening": continue
            if f["edge_idx"]==i and f["t_start"]<0.01:
                if any(f2["fixture_type"]=="opening" and f2["edge_idx"]==prev and f2["t_end"]>0.99 for f2 in fixtures):
                    skip=True
            if f["edge_idx"]==prev and f["t_end"]>0.99:
                if any(f2["fixture_type"]=="opening" and f2["edge_idx"]==i and f2["t_start"]<0.01 for f2 in fixtures):
                    skip=True
        if not skip:
            lines.append((floor[i].copy(), ceil[i].copy(), EDGE_CLR, 1.0, 0.85))

    return faces, lines, labels


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def generate_isometric(img_bgr: np.ndarray) -> bytes:
    h_img, w_img = img_bgr.shape[:2]

    # 1. contour
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,4)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilated = cv2.dilate(thresh, kern, iterations=2)
    cnts,_ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: raise ValueError("No contours")
    cnt = max(cnts, key=cv2.contourArea)

    # 2. polygon
    pts_px = _ensure_ccw(_detect_polygon(cnt))
    n = len(pts_px)

    # 3. OCR
    ocr = _ocr_labels(img_bgr, w_img, h_img)
    meas = _assign_labels(pts_px, ocr)

    # 4. fixtures
    fixtures = _detect_colors(img_bgr, pts_px)
    print(f"[DEBUG] {len(fixtures)} fixtures:")
    for f in fixtures:
        print(f"  e{f['edge_idx']}: {f['fixture_type']} t=[{f['t_start']:.3f},{f['t_end']:.3f}]")

    # 5. scale
    epx = [_edge_len(pts_px,i) for i in range(n)]
    if meas:
        sc = float(np.median([v/epx[ei] for ei,(v,_) in meas.items() if epx[ei]>0]))
    else:
        sc = 4.0 / max(epx)
    for i in range(n):
        if i not in meas: meas[i] = (epx[i]*sc, f"{epx[i]*sc:.2f} m")

    # 6. 3D
    pm = pts_px * sc
    pm[:,0] -= pm[:,0].min(); pm[:,1] -= pm[:,1].min()
    fl = np.column_stack([pm[:,0], pm[:,1], np.zeros(n)])
    cl = np.column_stack([pm[:,0], pm[:,1], np.full(n, WALL_H)])

    # 7. camera
    er, ar = np.radians(CAM_ELEV), np.radians(CAM_AZIM)
    cam = np.array([np.cos(er)*np.cos(ar), np.cos(er)*np.sin(ar), np.sin(er)])

    # 8. build scene
    faces, lines, labels = _build_scene(n, fl, cl, pm, fixtures, cam)

    # 9. sort faces: painter's algorithm
    faces.sort(key=lambda f: _face_depth(f[0], cam), reverse=True)

    # 10. render
    sx, sy = np.ptp(pm[:,0]), np.ptp(pm[:,1])
    fig = plt.figure(figsize=(13,9), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG); ax.grid(False); ax.set_axis_off()
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill=False; p.set_edgecolor("none")

    # floor grid
    path2d = MplPath(pm)
    for x in np.arange(0, sx+0.5, 0.5):
        sg = [(x,y) for y in np.arange(0,sy+0.5,0.05) if path2d.contains_point((x,y))]
        if len(sg)>=2:
            ax.plot([sg[0][0],sg[-1][0]],[sg[0][1],sg[-1][1]],[.001,.001],
                    color=FLOOR_EDGE,linewidth=0.35,alpha=0.45)
    for y in np.arange(0, sy+0.5, 0.5):
        sg = [(x,y) for x in np.arange(0,sx+0.5,0.05) if path2d.contains_point((x,y))]
        if len(sg)>=2:
            ax.plot([sg[0][0],sg[-1][0]],[sg[0][1],sg[-1][1]],[.001,.001],
                    color=FLOOR_EDGE,linewidth=0.35,alpha=0.45)

    # paint faces
    for verts, fc, ec, alpha, lw in faces:
        ax.add_collection3d(Poly3DCollection(
            [[np.array(v).tolist() for v in verts]],
            facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=lw))

    # paint lines
    for p0,p1,c,lw,a in lines:
        ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]], color=c, linewidth=lw, alpha=a)

    # measurement labels
    for i,(_, lbl) in meas.items():
        j=(i+1)%n; mid=(cl[i]+cl[j])/2; nrm=_outward_normal(pm,i)
        off=np.array([nrm[0],nrm[1],0])*0.25
        ax.text(mid[0]+off[0],mid[1]+off[1],mid[2]+0.15, lbl,
                fontsize=7.5, fontweight="bold", color="#222", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25",facecolor="white",edgecolor="#aaa",alpha=0.95,linewidth=0.6),
                zorder=50)

    # fixture labels
    for x,y,z,txt,fs,bfc,bec in labels:
        ax.text(x,y,z,txt, fontsize=fs, fontweight="bold", color="#111", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2",facecolor=bfc,edgecolor=bec,alpha=0.9,linewidth=0.5),
                zorder=55)

    # limits
    pad=0.5
    ax.set_xlim(pm[:,0].min()-pad, pm[:,0].max()+pad)
    ax.set_ylim(pm[:,1].min()-pad, pm[:,1].max()+pad)
    ax.set_zlim(-0.3, WALL_H+pad)
    ax.set_box_aspect([max(sx,0.1), max(sy,0.1), WALL_H])
    ax.view_init(elev=CAM_ELEV, azim=CAM_AZIM)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
