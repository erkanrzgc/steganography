"""Malicious graph headers/shapes cannot reach native ONNX allocations."""

import sys

import numpy as np
import pytest

from core import srnet_onnx_guard as guard

onnx = pytest.importorskip("onnx")


def graph(nodes=None):
    h = onnx.helper
    return h.make_model(
        h.make_graph(
            nodes
            or [
                h.make_node("GlobalAveragePool", ["pixels"], ["pooled"]),
                h.make_node("Flatten", ["pooled"], ["flat"], axis=1),
                h.make_node("MatMul", ["flat", "weights"], ["logits"]),
            ],
            "fixture",
            [h.make_tensor_value_info("pixels", onnx.TensorProto.FLOAT, ["batch", 1, 256, 256])],
            [h.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, ["batch", 2])],
            initializer=[onnx.numpy_helper.from_array(np.ones((1, 2), dtype="f4"), name="weights")],
        )
    )


def test_minimal_valid_graph_and_budget(monkeypatch):
    model = graph()
    guard.preflight(model.SerializeToString())
    monkeypatch.setattr(guard, "MAX_ACTIVATION_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        guard.preflight(model.SerializeToString())


def test_absent_optional_parser_and_unsafe_declared_output(monkeypatch):
    model = graph()
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 2**40
    with pytest.raises(ValueError, match="declared output"):
        guard.preflight(model.SerializeToString())
    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(RuntimeError, match="optional research"):
        guard.preflight(b"bounded")


@pytest.mark.parametrize(
    "fault",
    [
        "inputs",
        "shape",
        "type",
        "huge-initializer",
        "external",
        "domain",
        "unknown",
        "missing-input",
        "duplicate-output",
        "classifier",
        "pooling",
        "output",
        "duplicate-attribute",
        "conv-input",
        "conv-padding",
    ],
)
def test_graph_rejections(fault):
    model = graph()
    g = model.graph
    if fault == "inputs":
        g.input.extend([g.input[0]])
    if fault == "shape":
        g.input[0].type.tensor_type.shape.dim[2].dim_value = 2**40
    if fault == "type":
        g.input[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    if fault == "huge-initializer":
        g.initializer[0].dims[0] = 2**40
    if fault == "external":
        g.initializer[0].external_data.add(key="location", value="private")
    if fault == "domain":
        g.node[0].domain = "custom"
    if fault == "unknown":
        g.node[0].op_type = "ConstantOfShape"
    if fault == "missing-input":
        g.node[0].input[0] = "absent"
    if fault == "duplicate-output":
        g.node[0].output[0] = "pixels"
    if fault == "classifier":
        g.initializer[0].dims[0] = 2
        g.initializer[0].raw_data = np.ones((2, 2), dtype="f4").tobytes()
    if fault == "pooling":
        g.node[0].op_type = "AveragePool"
    if fault == "output":
        g.node[-1].output[0] = "other"
    if fault == "duplicate-attribute":
        g.node[1].attribute.extend([g.node[1].attribute[0]])
    if fault == "conv-input":
        g.node[0].op_type = "Conv"
    if fault == "conv-padding":
        g.initializer.extend(
            [onnx.numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype="f4"), name="kernel")]
        )
        g.node[0].CopyFrom(
            onnx.helper.make_node(
                "Conv", ["pixels", "kernel"], ["pooled"], kernel_shape=[1, 1], pads=[2**30] * 4
            )
        )
    with pytest.raises(ValueError):
        guard.preflight(model.SerializeToString())
