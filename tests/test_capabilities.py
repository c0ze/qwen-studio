from __future__ import annotations

from qwen_studio import moss
from qwen_studio.editing import capabilities


class InfoClient:
    def __init__(self, nodes):
        self.nodes = nodes

    def request_json(self, path, **kwargs):
        assert path == "/object_info"
        return self.nodes


def moss_nodes(unet=None, clip=None, vae=None, types=("qwen_image", "moss_image")):
    return {
        "UNETLoader": {"input": {"required": {"unet_name": [[moss.MODEL] if unet is None else unet]}}},
        "CLIPLoader": {"input": {"required": {
            "clip_name": [[moss.ENCODER] if clip is None else clip],
            "type": [list(types)],
        }}},
        "VAELoader": {"input": {"required": {"vae_name": [[moss.VAE] if vae is None else vae]}}},
    }


def test_moss_ready_when_files_and_node_support_exist():
    cap = capabilities(InfoClient(moss_nodes()))
    assert cap["moss"] is True
    assert cap["moss_reason"] == "Ready"


def test_moss_needs_a_comfyui_that_knows_the_model():
    cap = capabilities(InfoClient(moss_nodes(types=("qwen_image",))))
    assert cap["moss"] is False
    assert "Update ComfyUI" in cap["moss_reason"]


def test_moss_reports_the_missing_file():
    cap = capabilities(InfoClient(moss_nodes(clip=[])))
    assert cap["moss"] is False
    assert moss.ENCODER in cap["moss_reason"]


def test_moss_accepts_v3_combo_inputs():
    nodes = moss_nodes()
    nodes["UNETLoader"]["input"]["required"]["unet_name"] = ["COMBO", {"options": [moss.MODEL]}]
    assert capabilities(InfoClient(nodes))["moss"] is True


def test_offline_comfy_disables_moss():
    class Offline:
        def request_json(self, *args, **kwargs):
            raise OSError("down")

    cap = capabilities(Offline())
    assert cap["moss"] is False and cap["edit"] is False


def test_capabilities_report_whether_comfy_answered():
    assert capabilities(InfoClient(moss_nodes()))["online"] is True

    class Offline:
        def request_json(self, *args, **kwargs):
            raise OSError("down")

    assert capabilities(Offline())["online"] is False
