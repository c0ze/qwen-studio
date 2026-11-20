# Qwen Studio

A local image workspace for **Qwen-Image-2512**, **Qwen-Image-Edit-2511** and, optionally, **MOSS-Image**, **NitroFusion 2** and **Z-Image Turbo** with your own LoRAs, powered by ComfyUI. Includes an Omarchy custom app, desktop launcher, user service, and optional bar plugin.

- Create images with 13 editable style templates and exact, reusable seeds.
- Use your own Z-Image Turbo LoRAs (up to three, with strengths and trigger words), from text or by restyling a reference image.
- Organize assets into persistent project folders; import, move, rename folders, search, and download.
- Use up to three reference images for character-guided scenes and natural-language edits with Qwen Edit.
- Crop to another aspect ratio, fit within a new canvas, upscale with Lanczos, or extend with AI outpainting. Every operation saves a new asset.
- Permanent asset deletion removes the actual image and its associated metadata.
- Serial job queue with pending-job cancellation, browser reconnection, and local prompt drafts.

## Requirements

- Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a local ComfyUI installation.
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) and the models below.
- A GPU/CPU configuration capable of running the Qwen models. Quantized models still need substantial RAM and VRAM; low-VRAM ComfyUI offloading may be necessary.
- Omarchy is optional. The browser app works on other systems; the desktop installer targets Linux with a systemd user session.

ComfyUI and Studio must share the same filesystem. `COMFYUI_URL` points to that ComfyUI instance, not a remote machine with different input/output storage.

## Quick start

```bash
git clone https://github.com/c0ze/qwen-studio.git
cd qwen-studio
uv sync
uv run qwen-studio
```

Open [127.0.0.1:8790](http://127.0.0.1:8790). Start ComfyUI separately, or provide `~/ComfyUI/run-qwen-image.sh`; Studio invokes that launcher if generation finds ComfyUI offline.

### Models

| Location inside ComfyUI | Default file |
| --- | --- |
| `models/diffusion_models/` | `qwen-image-2512-Q4_K_S.gguf` |
| `models/text_encoders/` | `qwen_2.5_vl_7b_fp8_scaled.safetensors` |
| `models/vae/` | `qwen_image_vae.safetensors` |
| `models/diffusion_models/` (optional editing) | `qwen-image-edit-2511-Q4_K_S.gguf` |

The generation GGUF is available from [Unsloth](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF). Encoder/VAE files and native ComfyUI instructions are linked in the [ComfyUI Qwen guide](https://docs.comfy.org/tutorials/image/qwen/qwen-image).

To download the optional **12.4 GB** edit model with SHA-256 verification:

```bash
uv run python -m qwen_studio.models
```

This downloads only [Unsloth's Qwen Edit GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF); it reuses the encoder and VAE. Model weights are not included in this repository and retain their own licenses. Edit controls display readiness based on the running ComfyUI node/model list.

### MOSS-Image (optional)

[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) (OpenMOSS, Apache-2.0) is a second text-to-image model: a 5B DiT in the FLUX.2 latent space with a Qwen3.5-9B text encoder. It is strong at rendered text and readable signage. Choose **Create from text · MOSS Image** in the Workflow menu. It does not do reference edits or outpainting.

It needs a ComfyUI with native MOSS support, which landed on upstream master after v0.40.0 (`e087d8f6` and follow-ups). To download the **15.4 GB** of int8 files from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) with SHA-256 verification:

```bash
uv run python -m qwen_studio.models moss
```

| Location inside ComfyUI | File |
| --- | --- |
| `models/diffusion_models/` | `moss_image_sft_int8_convrot.safetensors` |
| `models/text_encoders/` | `qwen3.5_9b_int8_convrot.safetensors` |
| `models/vae/` | `flux2-vae_small.safetensors` |

Studio uses the upstream reference graph: 25 steps, CFG 2.5, `er_sde`, skip-layer guidance on layer 22 for the first half of sampling, and an empty negative prompt. Steps and the negative prompt are remembered per model, so switching between Qwen and MOSS restores each model's own values (or its defaults). On an RX 9060 XT (16 GB, ROCm 7.0, `--lowvram`) a 1024² image samples at about 1.4 s per step.

### NitroFusion 2 (optional)

[NitroFusion 2](https://huggingface.co/ChenDY/NitroFusion2) (Apache-2.0) is a 1–2 step distillation of Z-Image-Turbo, for fast drafts: try prompts and compositions here, then render the keepers with Qwen or MOSS. Choose **Fast draft · NitroFusion 2** in the Workflow menu. It does not do reference edits or outpainting, and it runs at CFG 1, so it ignores the negative prompt.

It runs through ComfyUI-GGUF (the same custom node as the Qwen GGUFs). To download the **12.2 GB** of files with SHA-256 verification:

```bash
uv run python -m qwen_studio.models nitro
```

| Location inside ComfyUI | File |
| --- | --- |
| `models/diffusion_models/` | `nf2-Q8_0.gguf` |
| `models/text_encoders/` | `Qwen3-4B-Q8_0.gguf` |
| `models/loras/` | `nf2-dedistill-lora.safetensors` |
| `models/vae/` | `ae.safetensors` (the FLUX.1 autoencoder) |

Studio follows the published GGUF workflows: one step uses euler at shift 3; two steps use shift 6, with the de-distillation LoRA at 0.5 on the first step only, which removes some one-step artifacts. On an RX 9060 XT a 1024² image takes about 4 s at one step and 8 s at two, once the models are loaded (the first image after a load takes about 80 s). Smaller tiers, from `nf2-Q2_K_4XS.gguf` (2.0 GB) up, can be installed by hand and selected with `NITRO_MODEL`.

### Z-Image Turbo and your LoRAs (optional)

[Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Tongyi-MAI, Apache-2.0) is a 6B text-to-image model that samples in about 8 steps at CFG 1; NitroFusion 2 is a distillation of it, so it reuses NitroFusion's encoder and autoencoder. It is the model for **your own LoRAs**: anything trained on Z-Image Turbo (for example with ai-toolkit) works. Choose **Create from text · Z-Image Turbo + LoRAs** or **Restyle a reference · Z-Image Turbo + LoRAs** in the Workflow menu. Like NitroFusion it runs at CFG 1 and ignores the negative prompt.

| Location inside ComfyUI | File |
| --- | --- |
| `models/diffusion_models/` | `z_image_turbo_int8_convrot.safetensors` (or another Z-Image Turbo checkpoint via `ZIMAGE_MODEL`, loaded with ComfyUI's `UNETLoader`) |
| `models/text_encoders/` | `Qwen3-4B-Q8_0.gguf` (shared with NitroFusion 2; needs ComfyUI-GGUF) |
| `models/vae/` | `ae.safetensors` (shared with NitroFusion 2) |
| `models/loras/` | your LoRAs |

The model isn't downloaded by `qwen_studio.models`; install it by hand. The int8 checkpoint needs a ComfyUI recent enough to load int8 "convrot" weights (the same support MOSS-Image needs).

**LoRAs.** The LoRA panel lists every file ComfyUI offers in `models/loras` (except the NitroFusion de-distillation LoRA Studio applies itself), with the base model and trigger words read from the safetensors header: ai-toolkit and kohya write `ss_base_model_version` and `ss_tag_frequency`, and a `trigger_word` field is read too. Without metadata the base comes from the file name (`…_zimage_turbo_…`). Only LoRAs trained on Z-Image are offered for Z-Image; the API refuses others and unknown names. Pick up to three, each with a strength from −2 to 2; **Add to prompt** inserts a LoRA's trigger words. The LoRAs and strengths are saved with each image (sidecar JSON, shown in the asset view and kept by **Reuse settings**), so you can always tell which LoRA made a picture.

**Restyle** takes one reference image, scales and center-crops it to the chosen size, and denoises it only part of the way: **Restyle strength** 0.3–0.5 keeps the reference's shapes and colors closely, 0.6 is a good start, and 0.8–1 follows the prompt and LoRAs more freely. It suits turning a game render, a sketch or a blockout into concept art in a LoRA's style. Qwen Edit can't load Z-Image LoRAs; to edit and then restyle, run Reference / edit first, then restyle its result.

On an RX 9060 XT (16 GB, ROCm, `--lowvram`) 8 steps at 1024² sample in about 6 s (1.1–1.3 it/s), with or without LoRAs. A whole job takes 25–50 s, most of it loading the text encoder and model, because Studio unloads models when its queue drains; the first job after ComfyUI starts takes about 65 s. Tested 2026-11-20 with an ai-toolkit Z-Image Turbo LoRA: text-to-image with and without it, and restyle of a game render at LoRA strengths 0.7 and 1.0 (restyle strength 0.6).

## Omarchy installation

```bash
./install.sh
```

Installs the Python app via `uv tool`, a desktop entry, a user service, and **Omarchy menu → Custom → Qwen Studio**. Existing menu entries and comments are preserved, and a menu backup is saved before changes. No root access or changes to packaged Omarchy files are needed.

The launcher starts the service on demand and opens the workspace as a web app. Configuration lives in `~/.config/qwen-studio/environment`. After editing it, run `systemctl --user restart qwen-studio.service`. To start at login, optionally run `systemctl --user enable qwen-studio.service`.

An optional Omarchy bar widget is provided as a standard plugin:

```bash
omarchy plugin add https://github.com/c0ze/qwen-studio.git --enable --yes
```

Install the app first; the widget launches `qwen-studio-open`. For updates, pull the repository and rerun `./install.sh`, then restart the service when no job is running. `omarchy plugin update c0ze.qwen-studio --yes` updates the optional widget checkout.

Remove desktop integration with `qwen-studio-install --uninstall`, then `uv tool uninstall qwen-studio`. Remove the optional widget with `omarchy plugin remove c0ze.qwen-studio --yes`. Images, models and configuration are retained.

## Projects and image tools

**New folder** creates a project. Choosing a folder filters the archive and selects it as the destination for new images. The **Save to project** selector can choose a different destination. Open an asset to move it to another folder or Unfiled. Renaming a project does not break asset URLs.

Assets remain normal files: unfiled images live under `~/ComfyUI/output/LMStudio/`, and projects live in `LMStudio/projects/<stable-id>/`. Each project has a `.project.json` display name. Keep these files when backing up or moving a library. Existing MCP-generated images appear in Unfiled automatically.

**Reference / edit** on an asset adds it to the prompt's reference list; you can also import PNG, JPEG, or WebP files (20 MB / 32 MP maximum). Use descriptions such as “Keep the character in Picture 1, change the coat to blue, and use the setting from Picture 2.” Qwen Edit guides likeness but does not guarantee identical character identity. The text-only 2512 model does not accept these references.

| Tool | Behavior |
| --- | --- |
| Reference / edit | Qwen Edit creates a new image from instructions and up to three references |
| Change aspect ratio | Center-crops and resamples to the chosen dimensions |
| Fit canvas | Fits the whole image and adds dark borders, without distortion |
| Upscale | Lanczos resampling; **not** neural super-resolution or invented detail |
| Extend canvas | Experimental masked Qwen Edit outpainting; generates the new border and composites the original pixels back in unchanged |
| Restyle | Z-Image Turbo redraws one reference with the prompt and LoRAs, keeping its composition (partial denoise) |

AI generation/edit dimensions are multiples of 64 between 512 and 1664. Extend requires both dimensions to be at least the source dimensions and at least one to increase. Crop, fit, and upscale support 64–4096 pixels per dimension. Upscale preserves aspect ratio. For a larger original, create a smaller fitted copy before AI extension.

Reference files are snapshotted when a job is queued, so moving/deleting the archive source afterwards does not break that queued job. Temporary copies are removed after completion or cancellation. Originals are never overwritten by image tools.

## Templates and controls

Anime, Cartoon, Realistic, Medieval, Ink pen, Stylized, Watercolor, Oil painting, Pixel art, Cinematic, Cyberpunk, Clay miniature, and 3D asset concept are text templates, not separate models or LoRAs (**3D asset concept** frames one whole object on a plain background, the input image-to-3D generators want). Style instructions are editable and separate from the subject. **No style** removes the style layer. **Preview final prompt** shows the complete request.

**Reuse settings** restores the saved composed prompt without applying the style twice. Seed **0** produces new variations; fixed seeds preserve all 63 bits. **Ctrl/⌘ + Enter** generates from the prompt form. Drafts stay in browser storage. Queue history lasts for the running server session; it survives closing/reloading the browser, but not a server restart. Only pending jobs can be cancelled, and graceful shutdown waits for active work.

## Configuration

| Environment variable | Default |
| --- | --- |
| `COMFYUI_DIR` | `~/ComfyUI` |
| `COMFYUI_URL` | `http://127.0.0.1:8188` |
| `QWEN_MODEL` | `qwen-image-2512-Q4_K_S.gguf` |
| `QWEN_EDIT_MODEL` | `qwen-image-edit-2511-Q4_K_S.gguf` |
| `MOSS_MODEL` | `moss_image_sft_int8_convrot.safetensors` |
| `NITRO_MODEL` | `nf2-Q8_0.gguf` |
| `ZIMAGE_MODEL` | `z_image_turbo_int8_convrot.safetensors` |
| `QWEN_STUDIO_HOST` | `127.0.0.1` |
| `QWEN_STUDIO_PORT` | `8790` |

This is a trusted local application, with no user authentication. Keep it on loopback; do not expose it publicly. Browser writes from foreign origins are rejected. Image/model processing is local; model setup downloads from Hugging Face. The UI currently loads optional fonts from Google Fonts, with local fallbacks.

## Development

```bash
uv sync --extra dev
uv run pytest -q
node --check qwen_studio/static/app.js
node --test tests/test_templates.mjs
uv build
```

Python tests use temporary galleries and a fake generation client; they do not touch your images or need model weights. The tests cover filesystem boundaries, projects, reference snapshots, workflow construction (including Z-Image, LoRA chaining and restyle), LoRA metadata and base checks, local transforms, queue handling, seed fidelity, and menu installation.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). [MIT license](LICENSE). This is an independent project, not an official Qwen, ComfyUI, or Omarchy release.
