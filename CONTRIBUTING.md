# Contributing

Use `uv sync --extra dev` and the checks in the README. Keep tests independent of live GPU services and use temporary directories for assets. Never commit generated images, user prompts, model weights, credentials, or machine-specific configuration.

For workflow changes, add graph-level tests and report the ComfyUI/model versions and any live tests performed. For filesystem changes, cover traversal, symlinks, collisions, deletion, and preserving originals. Image tools must create new assets.

Open a focused issue or pull request with the problem, change, and verification. Report security problems privately through GitHub's security reporting feature when available; do not post private asset data.
