"""Tests for a Proton backend that registers itself at runtime.

The fixture library stands in for a backend that ships separately from Proton:
it is built against the Proton library rather than into it, and registers itself
through `proton::registerBackend()` when the test asks it to.
"""

import ctypes
import pathlib

import pytest

import triton
from triton._C.libproton import proton as libproton

FIXTURE = pathlib.Path(triton.__file__).parent / "_C" / "libproton_test_runtime_backend.so"


@pytest.fixture(scope="module")
def runtime_backend():
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE.name} is only built when TRITON_BUILD_UT is set")
    library = ctypes.CDLL(str(FIXTURE))
    library.proton_test_register_runtime_backend.restype = ctypes.c_bool
    library.proton_test_register_runtime_backend.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.proton_test_claim_device_types_until_exhausted.restype = ctypes.c_int
    return library


def test_registration_is_visible_and_selectable(runtime_backend):
    before = libproton.get_available_profilers()
    assert "runtime_profiler" not in before

    assert runtime_backend.proton_test_register_runtime_backend(b"runtime_profiler", b"runtime_backend")

    # A registered profiler is appended, so every profiler known before keeps
    # its position.
    assert libproton.get_available_profilers() == before + ["runtime_profiler"]
    assert libproton.select_profiler_from_triton_backend("runtime_backend") == "runtime_profiler"


def test_duplicate_profiler_name_is_rejected(runtime_backend):
    assert runtime_backend.proton_test_register_runtime_backend(b"duplicate_profiler", b"duplicate_backend")
    before = libproton.get_available_profilers()

    # Registering a taken name again, or the name of a built-in profiler,
    # changes nothing, so loading a backend twice is harmless.
    assert not runtime_backend.proton_test_register_runtime_backend(b"duplicate_profiler", b"other_backend")
    assert not runtime_backend.proton_test_register_runtime_backend(b"cupti", b"other_backend")
    assert libproton.get_available_profilers() == before
    with pytest.raises(ValueError):
        libproton.select_profiler_from_triton_backend("other_backend")


def test_device_type_slots_run_out_rather_than_alias(runtime_backend):
    # The fixture checks every value it is handed: -1 means a value outside the
    # reserved range, a value handed out twice, or no end to allocation. Other
    # backends in the process may already hold slots, so the count is not fixed.
    assert runtime_backend.proton_test_claim_device_types_until_exhausted() >= 0
    assert runtime_backend.proton_test_claim_device_types_until_exhausted() == 0
