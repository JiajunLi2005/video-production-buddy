from __future__ import annotations

import jsonschema
import pytest
from pathlib import Path
from PIL import Image

from tools.base_tool import ToolResult
from tools.video.muapi_video import MuapiVideo


class _FakeResponse:
    def __init__(
        self,
        payload: dict | None = None,
        content: bytes = b"fake mp4",
        status_code: int = 200,
    ) -> None:
        self._payload = payload or {}
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def _fail_network(*_args, **_kwargs):
    raise AssertionError("network called before local input validation")


@pytest.fixture(autouse=True)
def _available_dependencies(monkeypatch):
    monkeypatch.setattr("tools.video.muapi_video.shutil.which", lambda command: command)


def _mock_valid_download(monkeypatch, content):
    monkeypatch.setattr(
        "tools.video.muapi_video.download_video", lambda _url, path: path.write_bytes(content)
    )
    monkeypatch.setattr(
        MuapiVideo, "_validate_download", lambda _self, path: {"file_size_bytes": path.stat().st_size}
    )


def test_muapi_video_declares_video_contract() -> None:
    tool = MuapiVideo()

    assert tool.name == "muapi_video"
    assert tool.provider == "muapi"
    assert tool.agent_skills == ["ai-video-gen"]
    assert tool.supports["local_image_input"] is True
    assert tool.capabilities == ["text_to_video", "image_to_video"]


def test_muapi_video_requires_project_output_path_before_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MUAPI_API_KEY", raising=False)
    monkeypatch.delenv("MU_API_KEY", raising=False)
    monkeypatch.setattr("requests.post", _fail_network)

    result = MuapiVideo().execute(
        {"prompt": "A product hero shot", "output_path": "muapi.mp4"}
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")


def test_muapi_video_image_to_video_requires_image_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUAPI_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", _fail_network)

    result = MuapiVideo().execute(
        {
            "prompt": "Animate the product frame",
            "operation": "image_to_video",
            "output_path": "projects/demo/assets/video/muapi.mp4",
        }
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert "image_to_video requires image_url" in (result.error or "")


def test_muapi_video_rejects_mismatched_model_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUAPI_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", _fail_network)

    result = MuapiVideo().execute(
        {
            "prompt": "A product hero shot",
            "model_variant": "seedance-2-image-to-video",
            "output_path": "projects/demo/assets/video/muapi.mp4",
        }
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert "supports image_to_video" in (result.error or "")


def test_muapi_video_text_to_video_success_matches_api_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MUAPI_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    _mock_valid_download(monkeypatch, b"fake mp4")
    captured: dict[str, object] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return _FakeResponse({"request_id": "request-1", "status": "processing"})

    def fake_get(url, **kwargs):
        if url.endswith("/predictions/request-1/result"):
            assert kwargs["headers"]["x-api-key"] == "test-key"
            return _FakeResponse(
                {
                    "status": "completed",
                    "content": [{"video_url": {"url": "https://cdn.example.test/out.mp4"}}],
                }
            )
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    output_path = "projects/demo/assets/video/muapi.mp4"
    result = MuapiVideo().execute(
        {
            "prompt": "A product hero shot",
            "duration": 8,
            "aspect_ratio": "16:9",
            "output_path": output_path,
        }
    )

    assert result.success, result.error
    assert captured["url"] == "https://api.muapi.ai/api/v1/seedance-2-text-to-video"
    assert captured["headers"] == {
        "x-api-key": "test-key",
        "Content-Type": "application/json",
    }
    assert captured["json"] == {
        "prompt": "A product hero shot",
        "duration": 8,
        "aspect_ratio": "16:9",
    }
    assert (tmp_path / output_path).read_bytes() == b"fake mp4"
    assert result.data["prediction_id"] == "request-1"
    assert result.data["output_path"] == output_path
    jsonschema.validate(instance=result.data, schema=MuapiVideo().output_schema)


def test_muapi_video_image_to_video_maps_single_image_to_images_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MU_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    _mock_valid_download(monkeypatch, b"fake i2v")
    captured: dict[str, object] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return _FakeResponse({"data": {"id": "request-2"}})

    def fake_get(url, **_kwargs):
        if url.endswith("/predictions/request-2/result"):
            return _FakeResponse({"status": "completed", "output": {"video": "https://cdn.example.test/i2v.mp4"}})
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    result = MuapiVideo().execute(
        {
            "prompt": "Animate the product frame",
            "operation": "image_to_video",
            "image_url": "https://example.test/product.png",
            "output_path": "projects/demo/assets/video/muapi-i2v.mp4",
        }
    )

    assert result.success, result.error
    assert captured["url"] == "https://api.muapi.ai/api/v1/seedance-2-image-to-video"
    assert captured["json"] == {
        "prompt": "Animate the product frame",
        "duration": 5,
        "aspect_ratio": "16:9",
        "images_list": ["https://example.test/product.png"],
    }
    assert (tmp_path / "projects/demo/assets/video/muapi-i2v.mp4").read_bytes() == b"fake i2v"


def test_muapi_video_rejects_insecure_output_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MUAPI_API_KEY", "test-key")

    def fake_post(_url, **_kwargs):
        return _FakeResponse({"request_id": "request-3"})

    def fake_get(url, **_kwargs):
        if url.endswith("/predictions/request-3/result"):
            return _FakeResponse({"status": "completed", "video_url": "http://example.test/out.mp4"})
        raise AssertionError("insecure output URL must not be downloaded")

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    result = MuapiVideo().execute(
        {
            "prompt": "A product hero shot",
            "output_path": "projects/demo/assets/video/muapi.mp4",
        }
    )

    assert not result.success
    assert "non-HTTPS" in (result.error or "")
    assert not (tmp_path / "projects/demo/assets/video/muapi.mp4").exists()


def test_muapi_video_is_discovered_by_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUAPI_API_KEY", "test-key")
    from tools.tool_registry import registry

    registry.clear()
    registry.discover()

    assert registry.get("muapi_video").get_status().value == "available"


def _mock_job(monkeypatch):
    monkeypatch.setenv("MUAPI_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", lambda *_a, **_kw: _FakeResponse({"request_id": "test-job"}))
    monkeypatch.setattr(
        MuapiVideo, "_poll_prediction",
        lambda *_a: {"video_url": "https://cdn.example.test/out.mp4"},
    )


@pytest.mark.parametrize("failure", ["invalid_media", "truncated_download"])
def test_failed_download_preserves_previous_artifact(monkeypatch, tmp_path, failure):
    monkeypatch.chdir(tmp_path)
    _mock_job(monkeypatch)
    output = Path("projects/demo/assets/video/out.mp4")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"previous approved video")

    def download(_url, candidate):
        assert candidate != output
        candidate.write_bytes(b"<html>upstream failure</html>")
        if failure == "truncated_download":
            raise OSError("truncated")

    def validate(_self, candidate):
        assert output.read_bytes() == b"previous approved video"
        assert candidate.read_bytes().startswith(b"<html>")
        raise ValueError("invalid video")

    monkeypatch.setattr("tools.video.muapi_video.download_video", download)
    monkeypatch.setattr(MuapiVideo, "_validate_download", validate)
    result = MuapiVideo().execute({"prompt": "test", "output_path": str(output)})
    assert not result.success
    assert not result.artifacts
    assert output.read_bytes() == b"previous approved video"
    assert list(output.parent.iterdir()) == [output]


def test_success_replaces_output_only_after_validation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _mock_job(monkeypatch)
    output = Path("projects/demo/assets/video/out.mp4")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"original")
    _mock_valid_download(monkeypatch, b"validated video")

    def validate(_self, candidate):
        assert output.read_bytes() == b"original"
        assert candidate.parent.resolve() == output.parent.resolve()
        return {"file_size_bytes": candidate.stat().st_size}

    monkeypatch.setattr(MuapiVideo, "_validate_download", validate)
    result = MuapiVideo().execute({"prompt": "test", "output_path": str(output)})
    assert result.success, result.error
    assert output.read_bytes() == b"validated video"
    assert list(output.parent.iterdir()) == [output]


def test_missing_decoder_fails_before_paid_request(monkeypatch):
    monkeypatch.setenv("MUAPI_API_KEY", "test-key")
    monkeypatch.setattr("tools.video.muapi_video.shutil.which", lambda _command: None)
    monkeypatch.setattr("requests.post", _fail_network)
    tool = MuapiVideo()
    assert tool.get_status().value != "available"
    result = tool.execute({"prompt": "test", "output_path": "projects/demo/assets/video/out.mp4"})
    assert not result.success
    assert "FFmpeg" in result.error


@pytest.mark.parametrize("image_bytes", [b"", b"not an image", b"x" * (10 * 1024 * 1024 + 1)], ids=["empty", "invalid", "oversized"])
def test_invalid_local_image_fails_before_upload(monkeypatch, tmp_path, image_bytes):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MUAPI_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", _fail_network)
    image = tmp_path / "input.png"
    image.write_bytes(image_bytes)
    result = MuapiVideo().execute({
        "prompt": "test", "operation": "image_to_video", "image_path": str(image),
        "output_path": "projects/demo/assets/video/out.mp4",
    })
    assert not result.success
    assert "Invalid MuAPI reference image" in result.error


def test_selector_uploads_local_image_directly_to_muapi(monkeypatch, tmp_path):
    from tools.video.video_selector import VideoSelector

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MUAPI_API_KEY", "test-key")
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
    image = tmp_path / "approved.png"
    Image.new("RGB", (16, 16), "blue").save(image)
    _mock_valid_download(monkeypatch, b"validated video")
    monkeypatch.setattr(MuapiVideo, "_poll_prediction", lambda *_a: {"video_url": "https://cdn.example.test/out.mp4"})
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        if url.endswith("/upload_file"):
            assert kwargs["headers"] == {"x-api-key": "test-key"}
            assert kwargs["files"]["file"] == ("reference.png", image.read_bytes(), "image/png")
            assert kwargs["allow_redirects"] is False
            return _FakeResponse({"url": "https://s3.amazonaws.com/muapi-assets/ref.png"})
        assert url == "https://api.muapi.ai/api/v1/seedance-2-image-to-video"
        assert kwargs["json"]["images_list"] == ["https://s3.amazonaws.com/muapi-assets/ref.png"]
        return _FakeResponse({"request_id": "test-job"})

    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("requests.get", _fail_network)
    selector = VideoSelector()
    monkeypatch.setattr(selector, "_providers", lambda: [MuapiVideo()])
    inputs = {
        "prompt": "Animate approved frame", "operation": "image_to_video",
        "reference_image_path": str(image), "preferred_provider": "muapi",
        "allowed_providers": ["muapi"], "output_path": "projects/demo/assets/video/out.mp4",
    }
    ranked = selector.execute({**inputs, "operation": "rank", "target_operation": "image_to_video"})
    assert ranked.success
    assert ranked.data["rankings"][0]["status"] == "available"
    assert not calls
    result = selector.execute(inputs)
    assert result.success, result.error
    assert calls == [
        "https://api.muapi.ai/api/v1/upload_file",
        "https://api.muapi.ai/api/v1/seedance-2-image-to-video",
    ]


@pytest.mark.parametrize("progress", ["", "frame=0\nprogress=end\n"])
def test_validator_requires_at_least_one_decoded_frame(monkeypatch, tmp_path, progress):
    import json
    from types import SimpleNamespace

    output = tmp_path / "empty-video.mp4"
    output.write_bytes(b"container with video headers")

    def run(_self, command, **_kwargs):
        assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
        assert command[command.index("-format_whitelist") + 1] == "mov"
        assert command[command.index("-enable_drefs") + 1] == "0"
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout=json.dumps({
                "streams": [{"codec_type": "video", "width": 320, "height": 180}],
                "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "5"},
            }), stderr="")
        return SimpleNamespace(stdout=progress, stderr="Output file is empty")

    monkeypatch.setattr(MuapiVideo, "run_command", run)
    with pytest.raises(ValueError, match="no decodable video frames"):
        MuapiVideo()._validate_download(output)
