# unified_processor.py
import json, math
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image
# Keep atlas-related utilities isolated so texture packing can evolve without touching animation math.
from atlas_utils import (
    decode_textures, extract_face_patches, build_compact_tile,
    pack_tiles_to_square_atlas, sanitize_name
)

# Display only positions the item in scenes; animation should not rely on display hacks anymore.
DISPLAY_SLOTS = [
    "thirdperson_righthand","thirdperson_lefthand",
    "firstperson_righthand","firstperson_lefthand",
    "gui","head","ground","fixed",
]

def to_float(v, d=0.0):
    try: return float(v)
    except: return d

def plus8_xz(v):
    # Vanilla models pivot around +8 on X/Z; aligning here keeps authored pivots consistent.
    if not isinstance(v, list) or len(v) != 3: return [8.0,8.0,8.0]
    return [to_float(v[0]) + 8.0, to_float(v[1]), to_float(v[2]) + 8.0]

def build_display(offset=(0,0,0), scale=(1,1,1), rotation=(0,0,0)):
    # Uniform rotation lets creators nudge the held/GUI pose without mixing with animation logic.
    rx, ry, rz = float(rotation[0]), float(rotation[1]), float(rotation[2])
    return {
        slot: {
            "rotation": [rx, ry, rz],
            "translation": [float(offset[0]), float(offset[1]), float(offset[2])],
            "scale": [float(scale[0]), float(scale[1]), float(scale[2])]
        } for slot in DISPLAY_SLOTS
    }

# --- Minimal linear algebra kept inline to avoid external deps --------------------------------
def deg2rad(x): return x * math.pi / 180.0
def rad2deg(x): return x * 180.0 / math.pi

def mat_identity():
    return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

def mat_mul(a,b):
    out=[[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j]=sum(a[i][k]*b[k][j] for k in range(4))
    return out

def mat_translate(tx,ty,tz):
    m=mat_identity(); m[0][3]=tx; m[1][3]=ty; m[2][3]=tz; return m

def mat_rx(ax):
    c,s = math.cos(deg2rad(ax)), math.sin(deg2rad(ax))
    return [[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]]

def mat_ry(ay):
    c,s = math.cos(deg2rad(ay)), math.sin(deg2rad(ay))
    return [[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]]

def mat_rz(az):
    c,s = math.cos(deg2rad(az)), math.sin(deg2rad(az))
    return [[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]]

def mat_euler_xyz(rx,ry,rz):
    # Follows three.js "XYZ" convention so exported models match editor expectations.
    return mat_mul(mat_mul(mat_rz(rz), mat_ry(ry)), mat_rx(rx))

def mat3_from4(m4):
    return [[m4[0][0],m4[0][1],m4[0][2]],
            [m4[1][0],m4[1][1],m4[1][2]],
            [m4[2][0],m4[2][1],m4[2][2]]]

def mat3_mul(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def mat3_mul_vec(m,v):
    return [m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],
            m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],
            m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]]

def mat3_sub(a,b):
    return [[a[i][j]-b[i][j] for j in range(3)] for i in range(3)]

def mat3_identity():
    return [[1,0,0],[0,1,0],[0,0,1]]

def mat3_transpose(m):
    return [[m[0][0],m[1][0],m[2][0]],
            [m[0][1],m[1][1],m[2][1]],
            [m[0][2],m[1][2],m[2][2]]]

def vec_add(a,b): return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
def vec_sub(a,b): return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def euler_from_matrix_xyz(R3):
    # Extract XYZ-order Euler in degrees; stable enough for vanilla per-element rotation fields.
    r11,r12,r13 = R3[0][0],R3[0][1],R3[0][2]
    r21,r22,r23 = R3[1][0],R3[1][1],R3[1][2]
    r31,r32,r33 = R3[2][0],R3[2][1],R3[2][2]
    ry = math.asin(max(-1.0, min(1.0, r13)))
    cy = math.cos(ry)
    if abs(cy) > 1e-6:
        rx = math.atan2(-r23, r33)
        rz = math.atan2(-r12, r11)
    else:
        rx = math.atan2(r21, r22)
        rz = 0.0
    return [rad2deg(rx), rad2deg(ry), rad2deg(rz)]

def normalize_rot(R3):
    # Remove implicit scale/shear so Euler extraction and pivot solving stay well-conditioned.
    out=[[0,0,0],[0,0,0],[0,0,0]]
    for i in range(3):
        n = math.sqrt(sum(R3[i][j]*R3[i][j] for j in range(3)))
        out[i] = [R3[i][j]/n for j in range(3)] if n else [1 if i==j else 0 for j in range(3)]
    return out

def solve_pivot_from_RT(R, T):
    # Convert "rotate R then translate T" into "rotate around a single point P": (I - R) P = T.
    # Using this P as the element rotation origin aligns visuals with group-animated motion.
    I_minus_R = mat3_sub(mat3_identity(), R)
    a,b,c = I_minus_R[0]; d,e,f = I_minus_R[1]; g,h,i = I_minus_R[2]
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    if abs(det) < 1e-7:
        return None
    inv = [
        [(e*i - f*h)/det, (c*h - b*i)/det, (b*f - c*e)/det],
        [(f*g - d*i)/det, (a*i - c*g)/det, (c*d - a*f)/det],
        [(d*h - e*g)/det, (b*g - a*h)/det, (a*e - b*d)/det],
    ]
    return [
        inv[0][0]*T[0] + inv[0][1]*T[1] + inv[0][2]*T[2],
        inv[1][0]*T[0] + inv[1][1]*T[1] + inv[1][2]*T[2],
        inv[2][0]*T[0] + inv[2][1]*T[1] + inv[2][2]*T[2],
    ]

# --- Blockbench structure helpers -------------------------------------------------------------
def build_indices(bb: dict):
    # Collect element/group maps and each element's group chain so we can evaluate hierarchical transforms.
    elements = bb.get("elements", []) or []
    outliner = bb.get("outliner", []) or []

    elements_by_uuid = {e.get("uuid"): e for e in elements if isinstance(e, dict) and e.get("uuid")}
    groups_by_uuid: Dict[str, dict] = {}

    def dfs(node):
        if isinstance(node, dict) and node.get("uuid"):
            groups_by_uuid[node["uuid"]] = node
            for ch in node.get("children", []):
                if isinstance(ch, dict): dfs(ch)
    for item in outliner or []:
        dfs(item)

    parent_of: Dict[str, str] = {}
    def dfs_chain(node, parent_uuid=None):
        if isinstance(node, dict) and node.get("uuid"):
            uid = node["uuid"]
            if parent_uuid: parent_of[uid] = parent_uuid
            for ch in node.get("children", []):
                if isinstance(ch, str):
                    if ch in elements_by_uuid or ch in groups_by_uuid:
                        parent_of[ch] = uid
                elif isinstance(ch, dict):
                    dfs_chain(ch, uid)
    for item in outliner or []:
        dfs_chain(item, None)

    elem_to_group_chain: Dict[str, List[str]] = {}
    for e_uuid in elements_by_uuid.keys():
        chain=[]
        cur = parent_of.get(e_uuid)
        while cur:
            chain.append(cur)
            cur = parent_of.get(cur)
        chain.reverse()
        elem_to_group_chain[e_uuid] = chain

    return elements_by_uuid, groups_by_uuid, elem_to_group_chain

# Linear interpolation is sufficient here; it keeps timing predictable with Minecraft's constant tick rate.
def eval_animator_channels(animator: dict, t: float):
    def collect_channel(name):
        ks=[]
        for kf in animator.get("keyframes", []):
            if kf.get("channel") != name: continue
            time = float(kf.get("time", 0))
            dps = kf.get("data_points") or []
            if dps:
                v=dps[0]
                ks.append((time, [to_float(v.get("x",0)), to_float(v.get("y",0)), to_float(v.get("z",0))]))
        ks.sort(key=lambda x:x[0])
        return ks

    def lerp(a,b,u): return [a[i] + (b[i]-a[i]) * u for i in range(3)]
    def eval_from_keys(keys, default):
        if not keys: return default[:]
        if t <= keys[0][0]: return keys[0][1][:]
        if t >= keys[-1][0]: return keys[-1][1][:]
        for i in range(len(keys)-1):
            t0,v0 = keys[i]; t1,v1 = keys[i+1]
            if t0 <= t <= t1:
                u = 0 if t1==t0 else (t - t0)/(t1 - t0)
                return lerp(v0, v1, u)
        return default[:]

    trans = eval_from_keys(collect_channel("position") or collect_channel("translation"), [0,0,0])
    rot   = eval_from_keys(collect_channel("rotation"), [0,0,0])
    scale = eval_from_keys(collect_channel("scale"), [1,1,1])  # scale is ignored downstream for stability
    return trans, rot, scale

def group_local_matrix_at_time(group: dict, anim: dict, t: float):
    # Build a group local transform around its pivot; stacking groups matches Blockbench's mental model.
    gid = group.get("uuid")
    animator = (anim.get("animators", {}) or {}).get(gid, {})
    trans, rot, _scl = eval_animator_channels(animator, t)

    ox, oy, oz = 0.0, 0.0, 0.0
    if isinstance(group.get("origin"), list) and len(group["origin"]) == 3:
        ox, oy, oz = [to_float(v) for v in group["origin"]]
    ox += 8.0; oz += 8.0  # align to vanilla +8 authoring convention

    M = mat_identity()
    M = mat_mul(M, mat_translate(trans[0], trans[1], trans[2]))
    M = mat_mul(M, mat_translate(ox, oy, oz))
    M = mat_mul(M, mat_euler_xyz(rot[0], rot[1], rot[2]))
    M = mat_mul(M, mat_translate(-ox, -oy, -oz))
    return M

def compose_group_matrix(chain: List[str], groups_by_uuid: Dict[str, dict], anim: dict, t: float):
    # Concatenate transforms so an element inherits all upstream group motion.
    M = mat_identity()
    for gid in chain:
        g = groups_by_uuid.get(gid)
        if not g: continue
        M = mat_mul(M, group_local_matrix_at_time(g, anim, t))
    return M

def build_models_for_animation(
    bb: dict,
    model_name: str,
    anim: dict,
    fps: int,
    png_base: Path,
    json_base: Path,
    display_offset = (0.0, 0.0, 0.0),
    display_scale  = (1.0, 1.0, 1.0),
    display_rotation = (0.0, 0.0, 0.0),  # NEW
):
    """
    Exports one vanilla model per sampled frame. Each model uses element-level XYZ rotation and a shared 2-frame texture:
    frame 0 holds the atlas; frame 1 is transparent. A per-image .mcmeta 'frames' array shows atlas only on that frame's
    global index, so texture size stays fixed regardless of animation length.
    """

    # Frame sampling pairs with .mcmeta timing; compositing lets vanilla swap frames without code.
    elements_by_uuid, groups_by_uuid, elem_to_chain = build_indices(bb)
    length = float(anim.get("length", 0) or 0.0)
    step = 1.0 / max(1, fps)
    times: List[float] = []
    tcur = 0.0
    while tcur < length - 1e-9:
        times.append(round(tcur, 6)); tcur += step
    if not times: times = [0.0]
    TOTAL_FRAMES = len(times)
    mc_frametime = max(1, int(round(20.0 / max(1, fps))))

    # One atlas for all elements avoids duplicating pixels per frame; the gating lives in .mcmeta.
    tex_images, uv_size = decode_textures(bb)
    tiles: List[Tuple[str, Image.Image, Dict[str, List[int]]]] = []
    face_uv_local_by_uuid: Dict[str, Dict[str, List[int]]] = {}
    for uuid, el in elements_by_uuid.items():
        patches = extract_face_patches(el, tex_images, uv_size)
        if patches:
            tile_img, face_uv_local, _ = build_compact_tile(patches)
        else:
            tile_img = Image.new("RGBA", (16,16), (0,0,0,0))
            face_uv_local = {}
        tiles.append((uuid, tile_img, face_uv_local))
        face_uv_local_by_uuid[uuid] = face_uv_local

    atlas_img, placements, atlas_side = pack_tiles_to_square_atlas(tiles)
    atlas_scale = atlas_side / 16.0

    model_safe = sanitize_name(model_name)
    anim_safe  = sanitize_name(anim.get("name") or anim.get("uuid") or "default")
    png_dir  = (png_base  / model_safe / anim_safe);  png_dir.mkdir(parents=True, exist_ok=True)
    json_dir = (json_base / model_safe / anim_safe);  json_dir.mkdir(parents=True, exist_ok=True)

    # Two-frame spritesheet keeps texture height constant, independent of animation length.
    aw, ah = atlas_img.size
    TWO_FRAME_H = ah * 2

    model_ids: List[str] = []

    for fi, tval in enumerate(times):
        # Two-frame image: atlas on frame 0; frame 1 left blank.
        sheet = Image.new("RGBA", (aw, TWO_FRAME_H), (0, 0, 0, 0))
        sheet.paste(atlas_img, (0, 0))
        stem = f"frame_{fi+1:03d}"
        png_path  = png_dir  / f"{stem}.png"
        json_path = json_dir / f"{stem}.json"
        sheet.save(png_path)

        # Frames array gates visibility: only this frame index uses atlas (0), others use transparent (1).
        frames_seq = [1] * TOTAL_FRAMES
        frames_seq[fi] = 0
        (png_dir / f"{stem}.png.mcmeta").write_text(
            json.dumps({"animation": {"frametime": mc_frametime, "frames": frames_seq}}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        textures_value = "item/" + png_path.relative_to(png_base).as_posix()[:-4]

        elements_out = []
        for uuid, el in elements_by_uuid.items():
            chain = elem_to_chain.get(uuid, [])

            # Use the pure rotation basis of the concatenated group transform; ignore scale to keep Euler extraction sane.
            M_group = compose_group_matrix(chain, groups_by_uuid, anim, tval)
            R_b = normalize_rot(mat3_from4(M_group))
            T_total = [M_group[0][3], M_group[1][3], M_group[2][3]]

            # Find a single pivot that explains the group's rotate+translate, so the model can rotate about that point.
            P_body = solve_pivot_from_RT(R_b, T_total)
            if P_body is None:
                # Degenerate case (almost no rotation): fall back to last group origin or element center to avoid jitter.
                if chain:
                    g_last = groups_by_uuid.get(chain[-1], {})
                    if isinstance(g_last.get("origin"), list) and len(g_last["origin"])==3:
                        P_body = plus8_xz(g_last["origin"])
                    else:
                        f0 = plus8_xz(el.get("from", [8,8,8])); t0 = plus8_xz(el.get("to", [8,8,8]))
                        P_body = [(f0[0]+t0[0])/2.0, (f0[1]+t0[1])/2.0, (f0[2]+t0[2])/2.0]
                else:
                    f0 = plus8_xz(el.get("from", [8,8,8])); t0 = plus8_xz(el.get("to", [8,8,8]))
                    P_body = [(f0[0]+t0[0])/2.0, (f0[1]+t0[1])/2.0, (f0[2]+t0[2])/2.0]

            # Normalize element geometry; if no element origin, use the box center to avoid bias.
            f = plus8_xz(el.get("from", [8,8,8]))
            t_ = plus8_xz(el.get("to",   [8,8,8]))
            O_e = plus8_xz(el["origin"]) if isinstance(el.get("origin"), list) and len(el["origin"])==3 \
                 else [(f[0]+t_[0])/2.0, (f[1]+t_[1])/2.0, (f[2]+t_[2])/2.0]

            # Compose element's static rotation with body rotation so the orientation matches authored intent.
            el_rot = el.get("rotation", [0,0,0])
            if isinstance(el_rot, dict):
                el_rot = [to_float(el_rot.get("x",0)), to_float(el_rot.get("y",0)), to_float(el_rot.get("z",0))]
            else:
                el_rot = [to_float(el_rot[0] if len(el_rot)>0 else 0),
                          to_float(el_rot[1] if len(el_rot)>1 else 0),
                          to_float(el_rot[2] if len(el_rot)>2 else 0)]
            R_e = normalize_rot(mat3_from4(mat_euler_xyz(el_rot[0], el_rot[1], el_rot[2])))
            R_total = mat3_mul(R_b, R_e)
            rot_xyz = euler_from_matrix_xyz(R_total)

            # Moving the pivot from element origin to body pivot introduces an offset; compensate by shifting geometry.
            R_b_inv = mat3_transpose(R_b)
            R_e_inv = mat3_transpose(R_e)
            T_residual = vec_sub(T_total, vec_sub(P_body, mat3_mul_vec(R_b, P_body)))  # zero ideally; keeps numerics stable
            term1 = vec_sub(mat3_mul_vec(R_e_inv, O_e), O_e)
            term2 = vec_sub(P_body, mat3_mul_vec(R_e_inv, P_body))
            term3 = mat3_mul_vec(mat3_mul(R_e_inv, R_b_inv), T_residual)
            D = vec_add(vec_add(term1, term2), term3)

            # Faces move with the cube; UVs only need atlas offsets and 16-based conversion.
            faces_new = {}
            (px, py) = placements.get(uuid, (0,0))
            for d, (x1,y1,x2,y2) in (face_uv_local_by_uuid.get(uuid) or {}).items():
                ax1, ay1, ax2, ay2 = px + x1, py + y1, px + x2, py + y2
                faces_new[d] = {"uv": [ax1/atlas_scale, ay1/atlas_scale, ax2/atlas_scale, ay2/atlas_scale], "texture": "#0"}

            elements_out.append({
                "uuid": uuid,
                "name": el.get("name"),
                "from": [f[0]+D[0], f[1]+D[1], f[2]+D[2]],
                "to":   [t_[0]+D[0], t_[1]+D[1], t_[2]+D[2]],
                "rotation": {"x": rot_xyz[0], "y": rot_xyz[1], "z": rot_xyz[2], "origin": P_body},
                "faces": faces_new
            })

        # Apply uniform display rotation/offset/scale after baking animation.
        display = build_display(display_offset, display_scale, display_rotation)
        out_json = {"credit": "Made with Blockbench and converted by bmaMC", "textures": {"0": textures_value}, "display": display, "elements": elements_out}
        json_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

        rel = json_path.relative_to(json_base).as_posix()
        model_ids.append(f"minecraft:item/{rel[:-5]}")

    return model_ids
