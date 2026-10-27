import pytest

from triton._C.libtriton import gluon_ir, ir
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language import _layouts


def test_shared_linear_layout_shape():
    assert ttgl.SharedLinearLayout([[0, 1], [0, 2]]).shape == [1, 4]
    assert ttgl.SharedLinearLayout([[3, 1], [1, 0], [0, 1]]).shape == [4, 2]
    assert ttgl.SharedLinearLayout([[4]]).shape == [8]


def test_layout_converter_unknown_encoding(monkeypatch):
    monkeypatch.setattr(_layouts, "_layout_converters", [])
    context = ir.context()
    ir.load_dialects(context)
    builder = gluon_ir.GluonOpBuilder(context)

    def make_tensor(encoding):
        return builder.create_poison(builder.get_distributed_ty(builder.get_float_ty(), [32], encoding))

    # A string attribute stands in for an encoding core has no Gluon layout for.
    unknown = make_tensor(builder.get_string_attr("unknown"))
    with pytest.raises(ValueError, match="Unhandled encoding encountered"):
        builder.get_gluon_layout_from_tensor(unknown)

    layout = ttgl.BlockedLayout([1], [32], [1], [0])
    calls = []

    def convert(attr):
        calls.append(attr)
        return layout

    _layouts.register_layout_converter(lambda attr: None)
    _layouts.register_layout_converter(convert)
    assert builder.get_gluon_layout_from_tensor(unknown) == layout
    assert len(calls) == 1

    # Encodings core knows never reach the converters.
    assert builder.get_gluon_layout_from_tensor(make_tensor(layout._to_ir(builder))) == layout
    assert len(calls) == 1
