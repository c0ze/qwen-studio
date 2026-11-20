# Plan: Z-Image Turbo, Arda's LoRAs and an agent-friendly concept step

Branch `zimage-loras`, opened 2026-11-20. Qwen Studio becomes the concept-image step of the game asset
pipeline (concept image → 3D model on fal → rig and texture), replacing fal's nano-banana. Agents drive it
through `~/projects/utils/asset-skills/skills/qwen-studio/qwen_concept.py` on the Mac mini (HTTP API over
`ssh cachyos`); that script and the `qwen-studio` skill follow whatever this branch settles on.

## Why

Every concept-art LoRA in `~/ComfyUI/models/loras` is a **Z-Image-Turbo** LoRA (rank 16, trained on Colab):
`gand_loopwright-toybox`, `gand_broomline`, `gand_clearfell`, `gand_pharmakon`, `amiga_*`, ... (37 files, two
checkpoints each). `z_image_turbo_int8_convrot.safetensors` was added to `models/diffusion_models` on Nov 19.
Qwen Studio today has no Z-Image model and no user LoRAs, and Qwen-Image / Qwen-Image-Edit can't load a
Z-Image LoRA. So none of the LoRAs can be used from the app or the API yet.

## Work

1. **Z-Image Turbo as a model** (`model: "zimage"`), next to `qwen`, `moss` and `nitro`.
   - Graph like `nitro.py` (same family: Qwen3-4B encoder as `lumina2`, `ae.safetensors`, AuraFlow
     sampling shift, CFG 1, zeroed negative), but the int8 checkpoint through the loader ComfyUI uses for
     `moss_image_sft_int8_convrot` (int8 convrot needs a ComfyUI recent enough; check `/object_info`),
     about 8 steps, `ZIMAGE_MODEL` env var like the others.
   - Capability check and reason string in `editing.capabilities`, like MOSS and Nitro.
2. **User LoRAs.**
   - `/api/capabilities` lists them: the `lora_name` options of `LoraLoaderModelOnly` from
     `/object_info`, minus the internal `nf2-dedistill-lora`, each with its base model (from the
     safetensors metadata, else the `_zimage_turbo_` part of the name) and trigger words if the metadata
     has them (kohya `ss_tag_frequency`, ai-toolkit `trigger_word`, ...).
   - `GenerateRequest.loras: [{name, strength}]`, at most 3, strength -2..2 (default 1). Refuse a LoRA
     whose base doesn't match the chosen model (Z-Image LoRAs on `zimage`; on `nitro` only if a test
     shows NF2 takes them well).
   - Chain `LoraLoaderModelOnly` nodes after the model loader, before the sampling shift.
   - The UI: a LoRA picker with strength sliders, remembered per project.
   - Record the LoRAs and strengths in the image's sidecar JSON and in `Job.as_dict()`, so a game can
     say which LoRA made each concept (games log AI provenance).
3. **Restyle a reference with a LoRA** (`operation: "restyle"` on `zimage`): one source image →
   VAE encode → KSampler with `denoise` 0.3–0.9 (new `GenerateRequest.denoise`, default 0.6). This is the
   step agents need most: redraw a game render or a Qwen-Edit cleanup in the LoRA's style while keeping
   its shape. Later, if denoise alone drifts too far: Z-Image's ControlNet union (canny / depth) to pin
   the silhouette.
4. **A "3D asset concept" template** next to the 12 styles: one whole object, centred, plain light
   background, no cast shadow, three-quarter view from slightly above, even lighting. 3D generators read
   the silhouette and colours straight off the concept.
5. **Agent-facing details.**
   - `Job.as_dict()` already reports the resolved seed; keep it so, and add `loras`, `denoise`, `model`.
   - Keep the API usable without a browser: no `Origin` header must stay allowed (it is today).
   - `/api/capabilities` stays cheap enough to call before every job.

## Done when

- `uv run pytest -q` passes, with graph-level tests for the Z-Image graph, LoRA chaining, base
  mismatch refusal, restyle, and the sidecar fields (CONTRIBUTING.md).
- A live run on the RX 9060 XT: one text-to-image and one restyle with `gand_loopwright-toybox` at two
  strengths, seconds per image noted in the README like the MOSS and Nitro sections.
- README: Z-Image and LoRA sections (files, env vars, how LoRAs are listed).
- `qwen_concept.py` in asset-skills gets `--lora NAME[:STRENGTH]` and `--denoise`, and the `qwen-studio`
  skill's "Calling it from an agent" section is updated; then one Loopwright concept is made end to end
  from the Mac mini.
