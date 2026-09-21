"""Local, non-destructive image operations. All results are new PNG assets."""
from __future__ import annotations

import io
import uuid
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 32_000_000
MAX_UPLOAD = 20 * 1024 * 1024


def load_image(source: Path | bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("image exceeds 32 megapixels")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("invalid image or image too large") from exc


def new_path(root: Path, operation: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / f"Studio-{operation}-{uuid.uuid4().hex}.png"


def transform(source: Path, root: Path, operation: str, width: int, height: int) -> Path:
    if not (64 <= width <= 4096 and 64 <= height <= 4096):
        raise ValueError("output dimensions must be from 64 to 4096")
    image = load_image(source)
    if operation == "crop":
        image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    elif operation == "fit":
        image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color=(18, 20, 24))
    elif operation == "upscale":
        if width < image.width or height < image.height:
            raise ValueError("upscale dimensions must not shrink the original")
        if abs(width / height - image.width / image.height) > 0.01:
            raise ValueError("upscale must preserve aspect ratio")
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    else:
        raise ValueError("unknown image operation")
    path = new_path(root, operation)
    image.save(path, format="PNG")
    return path


def outpaint_input(source: Path, destination: Path, width: int, height: int) -> tuple[int, int]:
    image = load_image(source)
    if width < image.width or height < image.height or (width, height) == image.size:
        raise ValueError("extend needs a canvas larger than the source without shrinking either side")
    left, top = (width - image.width) // 2, (height - image.height) // 2
    canvas = Image.new("RGBA", (width, height), (128, 128, 128, 0))
    canvas.paste(image.convert("RGBA"), (left, top))
    canvas.save(destination)
    return left, top
