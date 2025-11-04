"""
@forker: LaminarRainbow 
@title: FATLABEL
@nickname: FATLABEL
@description: A modified fork of FatLabels to add additional input features. 
"""

import torch
import numpy as np
import folder_paths
import sys
import subprocess
import threading
import locale
import pandas as pd
import os
import re
import io
import shutil
import tempfile
import zipfile
import urllib.parse
import urllib.request

comfy__path = os.path.dirname(folder_paths.__file__)
fatlabel__path = os.path.join(os.path.dirname(__file__))


def _list_node_fonts():
    fonts_dir = os.path.join(fatlabel__path, "fonts")
    results = []
    try:
        for root, _, files in os.walk(fonts_dir):
            for f in files:
                if f.lower().endswith((".ttf", ".otf", ".ttc", ".otc")):
                    rel = os.path.relpath(os.path.join(root, f), fonts_dir)
                    results.append(rel.replace(os.sep, "/"))
    except Exception:
        pass

    if not results:
        # Provide a sane default, even if missing on disk
        results = ["Bevan-Regular.ttf"]
    return sorted(results)

def handle_stream(stream, is_stdout):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')

    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else: 
            print(msg, end="", file=sys.stderr)


def process_wrap(cmd_str, cwd=None, handler=None):
    print(f"🏷️ FATLABEL (execute): {cmd_str} in '{cwd}'")
    process = subprocess.Popen(cmd_str, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    if handler is None:
        handler = handle_stream

    stdout_thread = threading.Thread(target=handler, args=(process.stdout, True))
    stderr_thread = threading.Thread(target=handler, args=(process.stderr, False))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()


pip_list = None


def get_installed_packages():
    global pip_list

    if pip_list is None:
        try:
            result = subprocess.check_output([sys.executable, '-m', 'pip', 'list'], universal_newlines=True)
            pip_list = set([line.split()[0].lower() for line in result.split('\n') if line.strip()])
        except subprocess.CalledProcessError as e:
            print(f"🏷️ FATLABEL (ComfyUI-Manager): Failed to retrieve the information of installed pip packages.")
            return set()
    
    return pip_list
    

def is_installed(name):
    name = name.strip()
    pattern = r'([^<>!=]+)([<>!=]=?)'
    match = re.search(pattern, name)
    
    if match:
        name = match.group(1)
        
    result = name.lower() in get_installed_packages()
    return result
    

def is_requirements_installed(file_path):
    print(f"req_path: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if not is_installed(line):
                    return False
                    
    return True

print(f"🏷️ FATLABEL (check dependencies)")

if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
    pip_install = [sys.executable, '-s', '-m', 'pip', 'install']
    mim_install = [sys.executable, '-s', '-m', 'mim', 'install']
else:
    pip_install = [sys.executable, '-m', 'pip', 'install']
    mim_install = [sys.executable, '-m', 'mim', 'install']

try:
    from PIL import Image, ImageDraw, ImageColor, ImageFont
except Exception:
    process_wrap(pip_install + ['Pillow'])

try:
    from freetype import *  # Import freetype-py
except Exception:
    process_wrap(pip_install + ['freetype-py'])

print(f"🏷️ FATLABEL (loading) : (v0.2.4)")

class BasicFatLabel:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                # Dropdown of fonts shipped with this node (from ./fonts)
                "font_name": (_list_node_fonts(),),
                "font_color_hex": ("STRING", {"default": "#888888", "multiline": False}),
                "background_color_hex": ("STRING", {"default": "#000000", "multiline": False}),
                "font_size": ("INT", {"default": 72, "min": 1}),  # Font size in pixels
                "kerning_value": ("FLOAT", {"default": 0.0}),  # New input for kerning
                "transparent_background": ("BOOLEAN", {"default": False}),
            },
            # Optional: allow manual override with a custom absolute path
            # Leave empty by default so dropdown is used unless explicitly overridden.
            "optional": {
                "font_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_basic_fatlabel"
    CATEGORY = "🏷️ FATLABEL (Basic)"

    def create_basic_fatlabel(
        self,
        text="",
        background_color_hex="#000000",
        font_color_hex="#888888",
        font_name=_list_node_fonts()[0],
        font_path="",
        font_size=72,
        kerning_value=0.0,
        transparent_background=False,
    ):
        # Resolve selected font: prefer custom path if it points to a file, otherwise use dropdown choice
        if font_path and isinstance(font_path, str) and os.path.isfile(font_path):
            selected_font_path = font_path
        else:
            # Allow subfolders selected via dropdown (entries use forward slashes)
            selected_font_path = os.path.join(fatlabel__path, "fonts", *font_name.split("/"))

        if not text:
            # If text is empty, return a placeholder canvas directly
            canvas_width, canvas_height = 40, 40  # Set desired dimensions for an empty canvas
            if transparent_background:
                canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
            else:
                bg_color = ImageColor.getcolor(background_color_hex, "RGBA")
                canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color[:3])
            image_tensor_out = torch.tensor(np.array(canvas) / 255.0, dtype=torch.float32).unsqueeze(0)
            return image_tensor_out,


def _slugify_family(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return s or "font"


class GoogleFontDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "family": ("STRING", {"default": "Inter"}),
                "variant_hint": ("STRING", {"default": "Regular"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("font_path", "font_name")
    FUNCTION = "download"
    CATEGORY = "??? FATLABEL (Fonts)"

    def download(self, family: str, variant_hint: str = "Regular", overwrite: bool = False):
        base_dir = os.path.join(fatlabel__path, "fonts")
        fam_slug = _slugify_family(family)
        dest_dir = os.path.join(base_dir, fam_slug)
        os.makedirs(base_dir, exist_ok=True)

        if overwrite and os.path.isdir(dest_dir):
            try:
                shutil.rmtree(dest_dir)
            except Exception as e:
                print(f"??? FATLABEL (fonts): failed to clear existing dir: {e}")

        os.makedirs(dest_dir, exist_ok=True)

        url = "https://fonts.google.com/download?family=" + urllib.parse.quote(family)
        print(f"??? FATLABEL (download): {family} -> {url}")

        extracted = []
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp_zip = os.path.join(td, "font.zip")
                with urllib.request.urlopen(url) as resp, open(tmp_zip, "wb") as f:
                    shutil.copyfileobj(resp, f)

                with zipfile.ZipFile(tmp_zip) as zf:
                    for zi in zf.infolist():
                        name = zi.filename
                        if not name.lower().endswith((".ttf", ".otf")):
                            continue
                        out_path = os.path.join(dest_dir, os.path.basename(name))
                        with zf.open(zi) as src, open(out_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted.append(out_path)
        except Exception as e:
            print(f"??? FATLABEL (download/extract) failed: {e}")
            return "", ""

        chosen = ""
        if extracted:
            vh = (variant_hint or "").lower()
            regulars = [p for p in extracted if re.search(r"regular", os.path.basename(p), re.I)]
            hinted = [p for p in extracted if vh and re.search(re.escape(vh), os.path.basename(p), re.I)]
            chosen = (regulars[:1] or hinted[:1] or extracted[:1])[0]

        if not chosen:
            return "", ""

        rel_for_dropdown = os.path.relpath(chosen, os.path.join(fatlabel__path, "fonts")).replace(os.sep, "/")
        return chosen, rel_for_dropdown

        font_color_rgba = ImageColor.getcolor(font_color_hex, "RGBA")
        fill_color = font_color_rgba if transparent_background else font_color_rgba[:3]

        # Initial font size and maximum attempts for fitting text
        current_font_size = font_size
        max_attempts = 10

        for _ in range(max_attempts):
            font = ImageFont.truetype(selected_font_path, current_font_size)

            # Calculate glyph widths and apply kerning between characters (Pillow 10+)
            # getsize was removed; use getlength for width per character
            glyph_widths = [font.getlength(ch) for ch in text]
            kerning_total = max(0, len(text) - 1) * kerning_value
            actual_text_width = sum(glyph_widths) + kerning_total

            # Calculate text height
            # Use getbbox to compute height (top/bottom of bounding box)
            bbox = font.getbbox(text)
            text_height = (bbox[3] - bbox[1]) if bbox else current_font_size

            # Create canvas with appropriate width and height (using integers)
            canvas_width = max(1, int(round(actual_text_width + 40)))
            canvas_height = max(1, int(round(text_height + 40)))
            if transparent_background:
                canvas_mode = "RGBA"
                canvas_color = (0, 0, 0, 0)
            else:
                canvas_mode = "RGB"
                canvas_color = ImageColor.getcolor(background_color_hex, "RGBA")[:3]

            canvas = Image.new(canvas_mode, (canvas_width, canvas_height), canvas_color)

            # Draw text with adjusted font size and kerning (using integers for coordinates)
            draw = ImageDraw.Draw(canvas)
            # Use baseline-aware positioning to avoid clipping at the bottom.
            # getbbox returns (left, top, right, bottom) relative to baseline.
            # To center visually, offset by bbox top so that the baseline is positioned correctly.
            x = (canvas_width - actual_text_width) / 2 - (bbox[0] if bbox else 0)
            y = int(round((canvas_height - text_height) / 2 - (bbox[1] if bbox else 0)))

            for ch, ch_width in zip(text, glyph_widths):
                draw.text((x, y), ch, fill=fill_color, font=font)
                x += ch_width + kerning_value

            # Convert to PyTorch tensor efficiently
            image_tensor_out = torch.tensor(np.array(canvas) / 255.0, dtype=torch.float32).unsqueeze(0)

            return image_tensor_out,

NODE_CLASS_MAPPINGS = {
    "🏷️ FATLABEL (Basic)": BasicFatLabel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BasicFatLabel": "🏷️ FATLABEL (Basic)",

}
