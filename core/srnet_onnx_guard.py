"""Bound ONNX allocations/graphs before native runtime initialization."""

from __future__ import annotations

import math

MAX_ACTIVATION_BYTES = 1024**3


def preflight(data):
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX preflight requires optional research dependency") from exc
    graph = onnx.load_model_from_string(data).graph
    if (
        len(graph.input) != 1
        or graph.input[0].name != "pixels"
        or len(graph.output) != 1
        or graph.output[0].name != "logits"
        or not 1 <= len(graph.node) <= 128
        or len(graph.initializer) > 128
        or graph.sparse_initializer
    ):
        raise ValueError("invalid bounded ONNX graph")
    shape = graph.input[0].type.tensor_type.shape.dim
    if (
        len(shape) != 4
        or [d.dim_value for d in shape[1:]] != [1, 256, 256]
        or not shape[0].dim_param
        or graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ):
        raise ValueError("invalid ONNX input contract")
    output = graph.output[0].type.tensor_type
    if (
        output.elem_type != onnx.TensorProto.FLOAT
        or len(output.shape.dim) != 2
        or output.shape.dim[0].dim_param != shape[0].dim_param
        or output.shape.dim[1].dim_value != 2
    ):
        raise ValueError("invalid ONNX declared output")
    sizes = {"pixels": (4, 1, 256, 256)}
    for tensor in graph.initializer:
        dims = tuple(tensor.dims)
        if (
            tensor.data_type != onnx.TensorProto.FLOAT
            or tensor.external_data
            or tensor.data_location
            or not 1 <= len(dims) <= 4
            or any(not 1 <= d <= 512 for d in dims)
            or len(tensor.raw_data) != math.prod(dims) * 4
            or tensor.name in sizes
        ):
            raise ValueError("unsafe ONNX initializer")
        sizes[tensor.name] = dims
    total = 0
    for node in graph.node:
        if (
            node.domain
            or len(node.output) != 1
            or node.output[0] in sizes
            or not node.input
            or any(name not in sizes for name in node.input)
        ):
            raise ValueError("unsafe ONNX node topology")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if len(attrs) != len(node.attribute):
            raise ValueError("duplicate ONNX attributes")
        x = sizes[node.input[0]]
        op = node.op_type
        if op == "Conv":
            if len(node.input) not in (2, 3) or len(x) != 4:
                raise ValueError("invalid ONNX convolution")
            w = sizes[node.input[1]]
            k, stride, pads = (
                attrs.get("kernel_shape"),
                attrs.get("strides", [1, 1]),
                attrs.get("pads", [0] * 4),
            )
            if (
                len(w) != 4
                or w[1] != x[1]
                or k not in ([1, 1], [3, 3])
                or tuple(k) != w[2:]
                or stride not in ([1, 1], [2, 2])
                or pads != [k[0] // 2] * 4
                or attrs.get("group", 1) != 1
                or attrs.get("dilations", [1, 1]) != [1, 1]
                or attrs.get("auto_pad", b"NOTSET") != b"NOTSET"
                or attrs.keys()
                - {"kernel_shape", "strides", "pads", "group", "dilations", "auto_pad"}
                or (len(node.input) == 3 and sizes[node.input[2]] != (w[0],))
            ):
                raise ValueError("unsafe ONNX convolution shape")
            result = (x[0], w[0], *((x[i + 2] - 1) // stride[i] + 1 for i in range(2)))
        elif (
            op in {"Relu", "Identity"}
            and not attrs
            and len(node.input) == 1
            or op == "Add"
            and not attrs
            and len(node.input) == 2
            and x == sizes[node.input[1]]
        ):
            result = x
        elif op == "AveragePool" and len(x) == 4 and len(node.input) == 1:
            if (
                attrs.get("kernel_shape") != [3, 3]
                or attrs.get("strides") != [2, 2]
                or attrs.get("pads") != [1, 1, 1, 1]
                or attrs.get("count_include_pad", 0) != 0
                or attrs.get("ceil_mode", 0) != 0
                or attrs.keys()
                - {"kernel_shape", "strides", "pads", "count_include_pad", "ceil_mode"}
            ):
                raise ValueError("unsafe ONNX pooling")
            result = (x[0], x[1], (x[2] + 1) // 2, (x[3] + 1) // 2)
        elif op == "GlobalAveragePool" and not attrs and len(x) == 4 and len(node.input) == 1:
            result = (x[0], x[1], 1, 1)
        elif op == "Flatten" and attrs == {"axis": 1} and len(node.input) == 1:
            result = (x[0], math.prod(x[1:]))
        elif op == "MatMul" and not attrs and len(node.input) == 2 and len(x) == 2:
            w = sizes[node.input[1]]
            if len(w) != 2 or x[1] != w[0]:
                raise ValueError("invalid ONNX classifier")
            result = (x[0], w[1])
        else:
            raise ValueError("unsupported ONNX operation")
        total += math.prod(result) * 4
        if total > MAX_ACTIVATION_BYTES:
            raise ValueError("ONNX activation budget exceeded")
        sizes[node.output[0]] = result
    if sizes.get("logits") != (4, 2):
        raise ValueError("invalid ONNX output contract")
