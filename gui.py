# gui.py
# Tk GUI for batching .bbmodel → Minecraft resource pack generation.

import json
import shutil
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from typing import List, Tuple

# generator API
from unified_processor import build_models_for_animation, sanitize_name


APP_TITLE = "bmaMC"


# ----------------------------- Utilities -----------------------------

def resource_path(name: str) -> Path:
    """Resolve a resource next to this script (works in normal run and PyInstaller)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))  # type: ignore
    return Path(base) / name

def write_pack_mcmeta(base_dir: Path, description: str, pack_format: int) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    mcmeta = {"pack": {"description": description, "pack_format": int(pack_format)}}
    (base_dir / "pack.mcmeta").write_text(json.dumps(mcmeta, ensure_ascii=False, indent=2), encoding="utf-8")

def write_items_composite(items_base: Path, model_name: str, anim_label: str, model_ids: List[str]) -> Path:
    target_dir = items_base / sanitize_name(model_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{sanitize_name(anim_label)}.json"
    data = {
        "model": {
            "type": "minecraft:composite",
            "models": [{"type": "minecraft:model", "model": mid} for mid in model_ids]
        }
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

def clear_folder_contents(folder: Path) -> None:
    if not folder.exists():
        return
    for child in folder.iterdir():
        if child.is_file() or child.is_symlink():
            try:
                child.unlink()
            except Exception:
                pass
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


# ----------------------------- Logging -----------------------------

def log_setup_colors(text: ScrolledText) -> None:
    text.tag_configure("INFO",    foreground="#1f2937")  # gray-800
    text.tag_configure("SUCCESS", foreground="#16a34a")  # green-600
    text.tag_configure("WARN",    foreground="#d97706")  # amber-600
    text.tag_configure("ERROR",   foreground="#dc2626")  # red-600
    text.tag_configure("HEADER",  foreground="#2563eb")  # blue-600
    text.tag_configure("DIM",     foreground="#6b7280")  # gray-500
    text.configure(state="disabled")

def log_write(text: ScrolledText, msg: str, level: str = "INFO") -> None:
    text.configure(state="normal")
    text.insert("end", msg.rstrip() + "\n", (level.upper(),))
    text.see("end")
    text.configure(state="disabled")
    text.update_idletasks()


# ----------------------------- Generation -----------------------------

def run_generation(
    log: ScrolledText,
    files: List[Path],
    out_base: Path,
    description: str,
    pack_format: int,
    fps: int,
    display_offset: Tuple[float, float, float],
    display_scale: Tuple[float, float, float],
    display_rotation: Tuple[float, float, float],
    clear_first: bool,
) -> None:
    if not files:
        messagebox.showwarning("No files", "Add one or more .bbmodel first.")
        return
    if not out_base:
        messagebox.showwarning("No output folder", "Choose an output resource pack folder.")
        return

    try:
        if clear_first:
            log_write(log, f"[CLEAN] {out_base}", "WARN")
            clear_folder_contents(out_base)

        write_pack_mcmeta(out_base, description, pack_format)
        log_write(log, f"[pack.mcmeta] description={description!r}, pack_format={pack_format}", "HEADER")

        # Standard resource pack layout
        png_base   = out_base / "assets" / "minecraft" / "textures" / "item"
        json_base  = out_base / "assets" / "minecraft" / "models"   / "item"
        items_base = out_base / "assets" / "minecraft" / "items"
        png_base.mkdir(parents=True, exist_ok=True)
        json_base.mkdir(parents=True, exist_ok=True)
        items_base.mkdir(parents=True, exist_ok=True)

        for fpath in files:
            try:
                log_write(log, f"[LOAD] {fpath}", "INFO")
                data = json.loads(Path(fpath).read_text(encoding="utf-8"))
                model_name = sanitize_name(Path(fpath).stem)

                animations = data.get("animations", []) or [{"name": "default", "uuid": "default", "length": 0, "animators": {}}]
                log_write(log, f"  └─ animations: {len(animations)}", "DIM")

                for anim in animations:
                    anim_label = sanitize_name(anim.get("name") or anim.get("uuid") or "default")

                    model_ids = build_models_for_animation(
                        bb=data,
                        model_name=model_name,
                        anim=anim,
                        fps=int(fps),
                        png_base=png_base,
                        json_base=json_base,
                        display_offset=list(display_offset),
                        display_scale=list(display_scale),
                        display_rotation=list(display_rotation),
                    )

                    items_json_path = write_items_composite(items_base, model_name, anim_label, model_ids)
                    log_write(log, f"  └─ [{anim_label}] models={len(model_ids)} → {items_json_path}", "SUCCESS")

            except Exception as sub_e:
                log_write(log, "  └─ ERROR: " + "".join(traceback.format_exception_only(type(sub_e), sub_e)).strip(), "ERROR")

        log_write(log, "[DONE] Generation complete. Load the output folder as a resource pack.", "SUCCESS")

    except Exception as e:
        log_write(log, "FATAL: " + "".join(traceback.format_exception_only(type(e), e)).strip(), "ERROR")
        messagebox.showerror("Generation failed", f"{e}")


# ----------------------------- GUI -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x680")
        self.minsize(960, 560)

        # Try to set window icon (Windows .ico). Ignore if missing or unsupported platform.
        try:
            self.iconbitmap("icon.ico")
        except Exception:
            pass

        # State
        self.files: List[Path] = []

        # Root grid
        self.columnconfigure(0, weight=1)  # left column expands
        self.columnconfigure(1, weight=0)  # right column fixed
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        right = ttk.Frame(self, padding=(0,10,10,10))
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")

        # ----- Left: files (top), log (bottom)
        left.rowconfigure(0, weight=0)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Files panel
        files_frame = ttk.Frame(left)
        files_frame.grid(row=0, column=0, sticky="nsew", pady=(0,8))
        files_frame.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(files_frame)
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(toolbar, text="Add…", command=self.on_add).grid(row=0, column=0, padx=(0,6))
        ttk.Button(toolbar, text="Remove Selected", command=self.on_remove_selected).grid(row=0, column=1, padx=6)
        ttk.Button(toolbar, text="Clear List", command=self.on_clear_list).grid(row=0, column=2, padx=6)

        self.listbox = tk.Listbox(files_frame, selectmode=tk.EXTENDED, height=5)
        self.listbox.grid(row=1, column=0, sticky="nsew", pady=(6,0))
        self.listbox.bind("<Delete>", lambda e: self.on_remove_selected())

        # Log panel
        log_frame = ttk.Frame(left)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = ScrolledText(log_frame, height=20, font=("Consolas", 10))
        self.log.grid(row=0, column=0, sticky="nsew")
        log_setup_colors(self.log)

        # ----- Right: settings
        right.columnconfigure(0, weight=1)
        settings = ttk.Frame(right)
        settings.grid(row=0, column=0, sticky="nsew")

        # Output folder
        ttk.Label(settings, text="Output folder").grid(row=0, column=0, sticky="w", padx=8, pady=(8,2))
        self.out_var = tk.StringVar()
        out_row = ttk.Frame(settings); out_row.grid(row=0, column=1, sticky="ew", padx=(0,8), pady=(8,2))
        out_row.columnconfigure(0, weight=1)
        ttk.Entry(out_row, textvariable=self.out_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(out_row, text="Browse…", command=self.on_browse_out).grid(row=0, column=1, padx=(6,0))

        # pack.mcmeta
        ttk.Label(settings, text="Description").grid(row=1, column=0, sticky="w", padx=8, pady=2)
        self.desc_var = tk.StringVar(value="")
        ttk.Entry(settings, textvariable=self.desc_var).grid(row=1, column=1, sticky="ew", padx=(0,8), pady=2)

        ttk.Label(settings, text="Pack format").grid(row=2, column=0, sticky="w", padx=8, pady=2)
        self.pack_var = tk.IntVar(value=73)
        ttk.Spinbox(settings, from_=1, to=999, textvariable=self.pack_var, width=8).grid(row=2, column=1, sticky="w", padx=(0,8), pady=2)

        # Display settings (defaults per request)
        disp = ttk.LabelFrame(settings, text="Display")
        disp.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(8,2))
        for c in range(7): disp.columnconfigure(c, weight=1)

        def vec_row(row, label, defaults):
            ttk.Label(disp, text=label).grid(row=row, column=0, sticky="w", padx=(6,4), pady=2)
            v = [tk.DoubleVar(value=defaults[i]) for i in range(3)]
            ttk.Entry(disp, textvariable=v[0], width=6).grid(row=row, column=1, padx=2, pady=2)
            ttk.Entry(disp, textvariable=v[1], width=6).grid(row=row, column=2, padx=2, pady=2)
            ttk.Entry(disp, textvariable=v[2], width=6).grid(row=row, column=3, padx=2, pady=2)
            return v

        # Defaults: offset=(0,0,0), scale=(1,1,1), rotation=(0,0,0)
        self.offset_vars = vec_row(0, "Offset (x y z)",  (0.0, 0.0, 0.0))
        self.scale_vars  = vec_row(1, "Scale  (x y z)",  (1.0, 1.0, 1.0))
        self.rot_vars    = vec_row(2, "Rotation (x y z)",(0.0, 0.0, 0.0))

        # FPS
        ttk.Label(settings, text="FPS").grid(row=4, column=0, sticky="w", padx=8, pady=6)
        self.fps_var = tk.IntVar(value=20)
        ttk.Spinbox(settings, from_=1, to=60, textvariable=self.fps_var, width=8).grid(row=4, column=1, sticky="w", padx=(0,8), pady=6)

        # Action buttons
        btns = ttk.Frame(settings)
        btns.grid(row=5, column=0, columnspan=2, sticky="e", padx=8, pady=(10,8))
        ttk.Button(btns, text="Clear Folder and Generate", command=self.on_generate_clean).grid(row=0, column=0, padx=6)
        ttk.Button(btns, text="Generate", command=self.on_generate).grid(row=0, column=1)

        # Allow right column to size itself naturally
        for r in range(6):
            settings.rowconfigure(r, weight=0)
        settings.columnconfigure(1, weight=1)

        # Welcome messages on first launch
        log_write(self.log, "Welcome to bmaMC, made by GodaOo", "HEADER")
        log_write(self.log, "Please use Blockbench to create Bedrock Entity models and animations. \nOnly translation and rotation are supported for now; scale is ignored.", "WARN")
        log_write(self.log, "-"*80, "INFO")

    # ----------------- Callbacks -----------------

    def on_add(self):
        paths = filedialog.askopenfilenames(
            title="Select .bbmodel",
            filetypes=[("Blockbench Model", "*.bbmodel"), ("All files", "*.*")]
        )
        if not paths:
            return
        added = 0
        for p in paths:
            p = Path(p)
            if p.exists() and p.suffix.lower() == ".bbmodel":
                if p not in self.files:
                    self.files.append(p)
                    self.listbox.insert("end", str(p))
                    added += 1
        if added:
            log_write(self.log, f"[ADD] {added} file(s) loaded.", "SUCCESS")

    def on_remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        sel.reverse()
        for idx in sel:
            try:
                del self.files[idx]
                self.listbox.delete(idx)
            except Exception:
                pass
        log_write(self.log, "[LIST] Removed selected.", "WARN")

    def on_clear_list(self):
        self.files.clear()
        self.listbox.delete(0, "end")
        log_write(self.log, "[LIST] Cleared.", "WARN")

    def on_browse_out(self):
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.out_var.set(path)

    def _read_vec(self, vars3) -> Tuple[float, float, float]:
        try:
            return (float(vars3[0].get()), float(vars3[1].get()), float(vars3[2].get()))
        except Exception:
            return (0.0, 0.0, 0.0)

    def on_generate(self):
        self._do_generate(clear_first=False)

    def on_generate_clean(self):
        out_base = Path(self.out_var.get().strip()) if self.out_var.get().strip() else None
        if out_base and out_base.exists():
            if not messagebox.askyesno("Confirm", f"This will remove all contents in:\n{out_base}\n\nContinue?"):
                return
        self._do_generate(clear_first=True)

    def _do_generate(self, clear_first: bool):
        out_base = Path(self.out_var.get().strip()) if self.out_var.get().strip() else None
        display_offset   = self._read_vec(self.offset_vars)
        display_scale    = self._read_vec(self.scale_vars)
        display_rotation = self._read_vec(self.rot_vars)

        try:
            run_generation(
                log=self.log,
                files=self.files.copy(),
                out_base=out_base,
                description=self.desc_var.get(),
                pack_format=int(self.pack_var.get()),
                fps=int(self.fps_var.get()),
                display_offset=display_offset,
                display_scale=display_scale,
                display_rotation=display_rotation,
                clear_first=clear_first,
            )
        except Exception as e:
            messagebox.showerror("Error", f"{e}")

if __name__ == "__main__":
    import sys
    App().mainloop()
