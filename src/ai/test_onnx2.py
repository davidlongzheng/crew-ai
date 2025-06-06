import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 10, 1, batch_first=False)
        self.lin1 = nn.Linear(10, 2)
        self.lin2 = nn.Linear(10, 2)

    def forward(self, x, h, c):
        x = x.unsqueeze(dim=0)
        x, (h, c) = self.lstm(x, (h, c))
        x = x.squeeze(dim=0)
        x = torch.where((x[:, 0] >= 1.0).unsqueeze(-1), self.lin1(x), self.lin2(x))

        return x, h, c


def run():
    model = TestModel()
    print([(k, v.shape) for k, v in model.named_parameters() if k.startswith("lstm")])

    batch_size = 1
    x = torch.rand((batch_size, 5))
    h = c = torch.zeros((1, batch_size, 10))

    inps = (
        x,
        h,
        c,
    )
    input_names = [
        "x",
        "h0",
        "c0",
    ]
    inp_batch_idxs = [0, 1, 1]
    output_names = ["out", "h1", "c1"]
    out_batch_idxs = [0, 1, 1]
    dynamic_axes = {}
    for name, batch_idx in zip(input_names, inp_batch_idxs):
        dynamic_axes[name] = {batch_idx: "B"}
    for name, batch_idx in zip(output_names, out_batch_idxs):
        dynamic_axes[name] = {batch_idx: "B"}
    print(dynamic_axes)

    with torch.no_grad():
        outs = model(*inps)

    print("outs", [x.shape for x in outs])

    model_fn = "my_model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            inps,
            model_fn,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
            report=True,
        )

    onnx_model = onnx.load(model_fn)
    onnx.checker.check_model(onnx_model)

    for t in onnx_model.graph.input:
        dims = [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
        print(t.name, dims)

    ort_session = onnxruntime.InferenceSession(model_fn)

    # ONNX expects numpy input
    new_batch_size = 2

    ort_inputs = {}
    for name, batch_idx, arr in zip(input_names, inp_batch_idxs, inps):
        tile = [1 for _ in range(len(arr.shape))]
        tile[batch_idx] = new_batch_size
        ort_inputs[name] = np.tile(arr.numpy(), tile)
    ort_outs = ort_session.run(None, ort_inputs)
    print("ort_outs", [x.shape for x in ort_outs])

    for out, ort_out, batch_idx in zip(outs, ort_outs, out_batch_idxs):
        out = np.take(out, 0, axis=batch_idx)
        ort_out = np.take(ort_out, 0, axis=batch_idx)
        assert np.isclose(out.numpy(), ort_out).all()
