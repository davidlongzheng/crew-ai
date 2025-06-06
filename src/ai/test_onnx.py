from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime
import torch
from tensordict import TensorDict
from torch import nn

import cpp_game
from ai.hyperparams import Hyperparams
from ai.models import PolicyValueModel, get_models
from game.settings import Settings


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim = 5
        self.out_dim = 3
        self.lins = nn.ModuleList(
            [nn.Linear(self.in_dim, self.out_dim) for _ in range(2)]
        )

    def forward(self, inps):
        k = inps["k"]
        x = inps["x"]

        return torch.where(k.unsqueeze(-1) == 0, self.lins[0](x), self.lins[1](x))


class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, k):
        return self.model(TensorDict(x=x, k=k))


def run():
    path = Path("/Users/davidzheng/projects/crew-ai/outdirs/0531/run_16")
    settings_dict = torch.load(path / "settings.pth", weights_only=False)
    hp = cast(Hyperparams, settings_dict["hp"])
    settings = cast(Settings, settings_dict["settings"])
    models = get_models(hp, settings, onnx=True)
    pv_model = cast(PolicyValueModel, models["pv"])
    checkpoint = torch.load(
        path / "checkpoint.pth",
        weights_only=False,
        map_location=torch.device("cpu"),
    )
    pv_model.load_state_dict(checkpoint["pv_model"])
    pv_model.eval()
    pv_model.start_single_step()

    model = pv_model.backbone_model
    assert not model.phase_branch

    batch_size = 5
    cpp_settings = settings.to_cpp()
    batch_rollout = cpp_game.BatchRollout(cpp_settings, batch_size)
    engine_seeds = list(range(batch_size))
    batch_rollout.reset_state(engine_seeds)
    move_inps = batch_rollout.get_move_inputs()

    h = c = torch.zeros((hp.hist_num_layers, batch_size, hp.hist_hidden_dim))

    inps = (
        torch.from_numpy(move_inps.hist_player_idx),
        torch.from_numpy(move_inps.hist_trick),
        torch.from_numpy(move_inps.hist_action),
        torch.from_numpy(move_inps.hist_turn),
        torch.from_numpy(move_inps.hist_phase),
        torch.from_numpy(move_inps.hand),
        torch.from_numpy(move_inps.player_idx),
        torch.from_numpy(move_inps.trick),
        torch.from_numpy(move_inps.turn),
        torch.from_numpy(move_inps.phase),
        torch.from_numpy(move_inps.task_idxs),
        h,
        c,
    )
    input_names = [
        "hist_player_idx",
        "hist_trick",
        "hist_action",
        "hist_turn",
        "hist_phase",
        "hand",
        "player_idx",
        "trick",
        "turn",
        "phase",
        "task_idxs",
        "h0",
        "c0",
    ]
    inp_batch_idxs = [0] * 11 + [1] * 2
    assert len(inp_batch_idxs) == len(inps) == len(input_names)
    output_names = ["out", "h1", "c1"]
    out_batch_idxs = [0, 1, 1]
    assert len(out_batch_idxs) == len(output_names)
    dynamic_axes = {}
    for name, batch_idx in zip(input_names, inp_batch_idxs):
        dynamic_axes[name] = {batch_idx: "B"}
    for name, batch_idx in zip(output_names, out_batch_idxs):
        dynamic_axes[name] = {batch_idx: "B"}
    print("dynamic_axes", dynamic_axes)

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
        arr = arr.numpy()
        if batch_idx == 0:
            arr = arr[:new_batch_size]
        else:
            assert batch_idx == 1
            arr = arr[:, :new_batch_size]
        ort_inputs[name] = arr
    ort_outs = ort_session.run(None, ort_inputs)
    print("ort_outs", [x.shape for x in ort_outs])

    for name, out, ort_out, batch_idx in zip(
        output_names, outs, ort_outs, out_batch_idxs
    ):
        out = out.numpy()
        if batch_idx == 0:
            out = out[:new_batch_size]
        else:
            assert batch_idx == 1
            out = out[:, :new_batch_size]
        print(name, out.shape, ort_out.shape)
        assert np.isclose(out, ort_out, atol=1e-6).all()
