"""Tests for how the instrumentation hook finds the backend of the active target."""

import pytest

import triton
import triton.backends
from triton.backends.compiler import GPUTarget
from triton.profiler.hooks.instrumentation import _get_backend_name


class _Compiler:

    def __init__(self, backend):
        self._backend = backend

    def supports_target(self, target):
        return target.backend == self._backend


class _Backend:

    def __init__(self, backend):
        self.compiler = _Compiler(backend)


class _Driver:

    def __init__(self, target):
        self._target = target

    def get_current_target(self):
        return self._target


@pytest.fixture
def active_target(monkeypatch):

    def install(backend, backends):
        target = GPUTarget(backend, "arch", 32)
        # `driver.active` is a read-only property that instantiates the real
        # driver on first use, so install the stand-in behind it.
        monkeypatch.setattr(triton.runtime.driver, "_active", _Driver(target), raising=False)
        monkeypatch.setattr(triton.backends, "backends",
                            {name: _Backend(name_backend)
                             for name, name_backend in backends.items()})

    return install


def test_in_tree_backends_keep_their_names(active_target):
    installed = {"nvidia": "cuda", "amd": "hip"}
    active_target("cuda", installed)
    assert _get_backend_name() == "nvidia"
    active_target("hip", installed)
    assert _get_backend_name() == "amd"


def test_backend_is_found_by_asking_the_backends(active_target):
    # A backend that is not one of the in-tree two is found the same way, which
    # is the point: the target name is not mapped here.
    active_target("other", {"nvidia": "cuda", "amd": "hip", "other_vendor": "other"})
    assert _get_backend_name() == "other_vendor"


def test_no_backend_for_the_target_is_an_error(active_target):
    active_target("other", {"nvidia": "cuda", "amd": "hip"})
    with pytest.raises(RuntimeError, match="found: \\[\\]"):
        _get_backend_name()


def test_ambiguous_backends_are_an_error(active_target):
    active_target("cuda", {"nvidia": "cuda", "impostor": "cuda"})
    with pytest.raises(RuntimeError, match="found: "):
        _get_backend_name()
