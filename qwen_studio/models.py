"""Download the optional Qwen Edit GGUF, verifying its published SHA-256."""
import hashlib
import urllib.request

from qwen_studio.comfy import COMFY_DIR

NAME = "qwen-image-edit-2511-Q4_K_S.gguf"
URL = f"https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/{NAME}"
SHA256 = "df952ef0d2b46463bd95d9afbb78e045ec5412316f453a7ad5a3d7bcbb111b72"


def main():
    destination = COMFY_DIR / "models/diffusion_models" / NAME
    if destination.exists():
        with destination.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() == SHA256:
                print(f"Already installed and verified: {destination}")
                return
        raise SystemExit("Existing model checksum differs; inspect it before replacing it.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.download')
    digest = hashlib.sha256()
    print(f"Downloading 12.4 GB to {destination}", flush=True)
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as handle:
            total = 0
            while block := response.read(8 * 1024 * 1024):
                handle.write(block); digest.update(block); total += len(block)
                print(f"\r{total / 1e9:.2f} GB downloaded", end='', flush=True)
        if digest.hexdigest() != SHA256:
            raise ValueError("Checksum verification failed; download discarded")
        partial.replace(destination)
        print("\nVerified and installed.")
    finally:
        partial.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
