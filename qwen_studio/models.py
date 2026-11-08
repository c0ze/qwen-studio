"""Download optional models, verifying each file's published SHA-256.

    python -m qwen_studio.models          # Qwen Edit 2511 GGUF (12.4 GB)
    python -m qwen_studio.models moss     # MOSS-Image SFT int8 + encoder + VAE (15.4 GB)
"""
import hashlib
import sys
import urllib.request

from qwen_studio import moss
from qwen_studio.comfy import COMFY_DIR

NAME = "qwen-image-edit-2511-Q4_K_S.gguf"
URL = f"https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/{NAME}"
SHA256 = "df952ef0d2b46463bd95d9afbb78e045ec5412316f453a7ad5a3d7bcbb111b72"

MOSS_URL = "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main"
# The SFT model is the default moss.MODEL; a different MOSS_MODEL is installed by hand.
MOSS_FILES = [
    ("diffusion_models", "moss_image_sft_int8_convrot.safetensors",
     "b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f"),
    ("text_encoders", moss.ENCODER, "32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a"),
    ("vae", moss.VAE, "d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b"),
]

BUNDLES = {
    "edit": [("diffusion_models", NAME, URL, SHA256)],
    "moss": [(folder, name, f"{MOSS_URL}/{folder}/{name}", sha) for folder, name, sha in MOSS_FILES],
}


def install(folder: str, name: str, url: str, sha256: str) -> None:
    destination = COMFY_DIR / "models" / folder / name
    if destination.exists():
        with destination.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() == sha256:
                print(f"Already installed and verified: {destination}")
                return
        raise SystemExit(f"Existing {name} checksum differs; inspect it before replacing it.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.download')
    digest = hashlib.sha256()
    print(f"Downloading {name} to {destination}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as handle:
            total = 0
            while block := response.read(8 * 1024 * 1024):
                handle.write(block); digest.update(block); total += len(block)
                print(f"\r{total / 1e9:.2f} GB downloaded", end='', flush=True)
        if digest.hexdigest() != sha256:
            raise ValueError("Checksum verification failed; download discarded")
        partial.replace(destination)
        print("\nVerified and installed.")
    finally:
        partial.unlink(missing_ok=True)


def main(argv: list[str] | None = None):
    bundle = (argv if argv is not None else sys.argv[1:]) or ["edit"]
    if len(bundle) != 1 or bundle[0] not in BUNDLES:
        raise SystemExit(f"usage: python -m qwen_studio.models [{'|'.join(BUNDLES)}]")
    for entry in BUNDLES[bundle[0]]:
        install(*entry)


if __name__ == '__main__':
    main()
