from __future__ import annotations

import jsonschema
import pytest

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


def test_muapi_video_declares_video_contract() -> None:
    tool = MuapiVideo()

    assert tool.name == "muapi_video"
    assert tool.provider == "muapi"
    assert tool.agent_skills == ["ai-video-gen"]
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
    monkeypatch.setattr(
        "tools.video.muapi_video.probe_output",
        lambda _path: {"file_size_bytes": 8},
    )
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
        if url == "https://cdn.example.test/out.mp4":
            assert kwargs["allow_redirects"] is False
            return _FakeResponse(content=b"fake mp4")
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
    monkeypatch.setattr(
        "tools.video.muapi_video.probe_output",
        lambda _path: {"file_size_bytes": 8},
    )
    captured: dict[str, object] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return _FakeResponse({"data": {"id": "request-2"}})

    def fake_get(url, **_kwargs):
        if url.endswith("/predictions/request-2/result"):
            return _FakeResponse({"status": "completed", "output": {"video": "https://cdn.example.test/i2v.mp4"}})
        if url == "https://cdn.example.test/i2v.mp4":
            return _FakeResponse(content=b"fake i2v")
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
