import json
import struct
import types

import pytest

import triton
from triton import _instrumentation
from triton.backends.compiler import GPUTarget
from triton.profiler.hooks import instrumentation
from triton.profiler.hooks.instrumentation import InstrumentationHook

TTGIR = """
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func public @kernel() {
    tt.return
  }
}
"""


class FakeDriver:

    def __init__(self, backend):
        self.backend = backend
        self.utils = types.SimpleNamespace(get_device_properties=lambda device: {"max_shared_mem": 1024})

    def get_current_target(self):
        return GPUTarget(self.backend, 0, 32)

    def get_current_device(self):
        return 0


@pytest.fixture
def fake_target(monkeypatch):
    """Makes `fake` the active target backend, with an instrumentation entry point that records its calls."""
    calls = []
    backend = types.SimpleNamespace(name="fake_backend", trace_device_type=7,
                                    lower_to_llvm=lambda pm, device: calls.append(("lower_to_llvm", pm, device)),
                                    load_dialects=lambda context: calls.append(("load_dialects", )))
    entry_point = types.SimpleNamespace(name="fake", load=lambda: backend)
    monkeypatch.setattr(instrumentation, "entry_points", lambda group: [entry_point]
                        if group == "triton.profiler.instrumentation" else [])
    monkeypatch.setattr(instrumentation, "libproton", types.SimpleNamespace(init_function_metadata=lambda *args: None))
    monkeypatch.setattr(triton.runtime.driver, "_active", FakeDriver("fake"))
    instrumentation._get_external_backend.cache_clear()
    yield calls
    instrumentation._get_external_backend.cache_clear()


def test_entry_point_backend(fake_target, tmp_path):
    (tmp_path / "kernel.ttgir").write_text(TTGIR)
    metadata = {"target": {"backend": "fake"}, "profile_scratch_size": 0, "num_warps": 1}
    (tmp_path / "kernel.json").write_text(json.dumps(metadata))
    hook = InstrumentationHook("")
    hook.activate()
    try:
        for point in ("load-dialects", "ttgpuir-to-llvmir", "llvmir-to-llvm"):
            assert (point, "fake_backend") in _instrumentation._callbacks
        _instrumentation._callbacks[("llvmir-to-llvm", "fake_backend")]("pm")
        files = {"kernel.ttgir": str(tmp_path / "kernel.ttgir"), "kernel.json": str(tmp_path / "kernel.json")}
        hook.init_handle(None, 1, "kernel", files, "")
        hook._populate_host_buffer(1)
        assert struct.unpack("6I", InstrumentationHook.host_buffer[:24].numpy().tobytes())[5] == 7
    finally:
        hook.deactivate()
    assert fake_target == [("lower_to_llvm", "pm", 0), ("load_dialects", )]
    assert not any(backend == "fake_backend" for _, backend in _instrumentation._callbacks)


def test_unsupported_backend(monkeypatch):
    monkeypatch.setattr(instrumentation, "entry_points", lambda group: [])
    monkeypatch.setattr(triton.runtime.driver, "_active", FakeDriver("unknown"))
    instrumentation._get_external_backend.cache_clear()
    with pytest.raises(RuntimeError, match="Unsupported backend: unknown"):
        InstrumentationHook("").activate()
    assert InstrumentationHook.active_count == 0
