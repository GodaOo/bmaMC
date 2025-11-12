# atlas_utils.py
import math, re, base64, io
from typing import Dict, List, Tuple
from PIL import Image

# Sanitize FS names so resource pack paths stay valid across platforms.
def sanitize_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(s or "noname")).strip() or "noname"

def parse_data_url(data_url: str):
    m = re.match(r"^data:(?P<mime>[^;]+);base64,(?P<data>.*)$", data_url or "", re.DOTALL)
    if not m: return None, None
    try:
        return m.group("mime").lower(), base64.b64decode(m.group("data"))
    except Exception:
        return None, None

def decode_textures(bb: dict):
    """
    Decodes embedded Blockbench textures once so all frames can reuse the same pixels.
    Sharing one atlas avoids duplicating identical image data per frame.
    """
    images: Dict[str, Image.Image] = {}
    uv_size: Dict[str, Tuple[int, int]] = {}
    for i, tex in enumerate(bb.get("textures", []) or []):
        mime, b = parse_data_url(tex.get("source", ""))
        if not b: continue
        img = Image.open(io.BytesIO(b)).convert("RGBA")
        key_idx = str(i); key_id = str(tex.get("id", i))
        uvw = int(tex.get("uv_width") or tex.get("width") or img.width)
        uvh = int(tex.get("uv_height") or tex.get("height") or img.height)
        for k in (key_idx, key_id):
            images[k]  = img
            uv_size[k] = (uvw, uvh)
    return images, uv_size

def _normalize_uv_rect(uv):
    # Keeps UV math predictable if authors swap corners.
    x1, y1, x2, y2 = map(float, uv)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]

def extract_face_patches(el: dict, tex_images, uv_size):
    """
    Crops per-face patches from the source texture using authored UVs.
    Doing this once enables compact tiling and consistent UV remapping later.
    """
    FACE_ORDER = ["north","east","south","west","up","down"]
    out = {}
    faces = el.get("faces", {}) or {}
    for d in FACE_ORDER:
        f = faces.get(d)
        if not isinstance(f, dict): continue
        uv = f.get("uv"); tex_key = str(f.get("texture","")).lstrip("#")
        img = tex_images.get(tex_key)
        if img is None or uv is None: continue
        x1,y1,x2,y2 = _normalize_uv_rect(uv)
        uvw, uvh = uv_size.get(tex_key, (img.width, img.height))
        sx = img.width / float(uvw) if uvw else 1.0
        sy = img.height / float(uvh) if uvh else 1.0
        L = math.floor(x1*sx); T = math.floor(y1*sy)
        R = math.ceil (x2*sx); B = math.ceil (y2*sy)
        L = max(0,min(L,img.width)); R=max(0,min(R,img.width))
        T = max(0,min(T,img.height));B=max(0,min(B,img.height))
        if R<=L or B<=T: continue
        patch = img.crop((L,T,R,B))
        # Preserve authoring intent if UVs were specified reversed.
        if float(uv[2]) < float(uv[0]): patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if float(uv[3]) < float(uv[1]): patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
        out[d] = (patch, (patch.width, patch.height))
    return out

def build_compact_tile(patches):
    """
    Packs up to two rows to keep tiles tight.
    Small tiles improve atlas packing density and reduce wasted transparent pixels.
    """
    if not patches: return Image.new("RGBA", (16,16), (0,0,0,0)), {}, (16,16)
    items = [(d, img.size[0], img.size[1], img) for d,(img,_sz) in patches.items()]
    items.sort(key=lambda x: x[2], reverse=True)  # taller first to balance rows
    row1,row2=[],[]
    w1=w2=h1=h2=0
    for d,w,h,img in items:
        if w1 <= w2:
            row1.append((d,w,h,img)); w1 += w; h1 = max(h1, h)
        else:
            row2.append((d,w,h,img)); w2 += w; h2 = max(h2, h)
    tile_w = max(w1, w2) if (row1 and row2) else (w1+w2)
    tile_h = (h1 + h2) if (row1 and row2) else max(h1, h2)
    tile = Image.new("RGBA", (tile_w, tile_h), (0,0,0,0))
    face_uv_local = {}
    x=0
    for d,w,h,img in row1:
        tile.paste(img, (x,0)); face_uv_local[d]=[x,0,x+w,h]; x+=w
    x=0; y=h1
    for d,w,h,img in row2:
        tile.paste(img, (x,y)); face_uv_local[d]=[x,y,x+w,y+h]; x+=w
    return tile, face_uv_local, (tile_w, tile_h)

def _ceil_to_multiple(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base

def _ceil_to_even16_multiple(x: int) -> int:
    # Even*16 avoids odd block rows that can waste a scanline in some packers.
    m16 = _ceil_to_multiple(max(1, x), 16)
    return m16 if (m16 // 16) % 2 == 0 else m16 + 16

def pack_tiles_to_square_atlas(tiles):
    """
    Simple skyline packer tuned for squared outputs.
    A square atlas keeps UV scaling uniform and is friendly to older resource pack workflows.
    """
    if not tiles:
        side = 32
    else:
        area = sum(t.size[0]*t.size[1] for _, t, _ in tiles)
        side = _ceil_to_even16_multiple(int(math.ceil(math.sqrt(area) * 1.2)))

    while True:
        x=y=row_h=0
        placements={}
        ok=True
        atlas = Image.new("RGBA", (side, side), (0,0,0,0))
        for uid, img, _ in sorted(tiles, key=lambda it: it[1].size[1], reverse=True):
            w,h = img.size
            if w>side or h>side: ok=False; break
            if x+w > side: x=0; y+=row_h; row_h=0
            if y+h > side: ok=False; break
            atlas.paste(img, (x,y))
            placements[uid]=(x,y)
            x += w
            row_h = max(row_h, h)
        if ok: return atlas, placements, side
        side += 32  # modest growth to avoid over-shooting and keep memory use predictable
