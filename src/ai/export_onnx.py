"""
ONNX export utility for Crew AI models.

This script exports a trained PolicyValueModel to ONNX format with validation.
Replicates the functionality of the gen_model() function with user-specified paths.
"""

from pathlib import Path

import click
import numpy as np
import onnx
import torch
from tensordict import TensorDict

import cpp_game
from ai.ai import load_ort_model
from ai.models import load_model_for_eval


def validate_model(model_path):
    click.echo("Validating")
    model, settings, hp = load_model_for_eval(model_path, onnx=False)
    ort_model = load_ort_model(model_path)

    batch_size = 5
    cpp_settings = settings.to_cpp()
    batch_rollout = cpp_game.BatchRollout(cpp_settings, batch_size)
    engine_seeds = list(range(batch_size))
    batch_rollout.reset_state(engine_seeds)

    h = c = np.zeros(
        (hp.hist_num_layers, batch_size, hp.hist_hidden_dim), dtype=np.float32
    )

    i = 0
    while not batch_rollout.is_done():
        move_inps = batch_rollout.get_move_inputs()

        inps = TensorDict(
            hist=TensorDict(
                player_idx=move_inps.hist_player_idx,
                trick=move_inps.hist_trick,
                action=move_inps.hist_action,
                turn=move_inps.hist_turn,
                phase=move_inps.hist_phase,
            ),
            private=TensorDict(
                hand=move_inps.hand,
                player_idx=move_inps.player_idx,
                trick=move_inps.trick,
                turn=move_inps.turn,
                phase=move_inps.phase,
                task_idxs=move_inps.task_idxs,
            ),
            valid_actions=move_inps.valid_actions,
        )
        ort_inps = {
            "hist_player_idx": move_inps.hist_player_idx,
            "hist_trick": move_inps.hist_trick,
            "hist_action": move_inps.hist_action,
            "hist_turn": move_inps.hist_turn,
            "hist_phase": move_inps.hist_phase,
            "hand": move_inps.hand,
            "player_idx": move_inps.player_idx,
            "trick": move_inps.trick,
            "turn": move_inps.turn,
            "phase": move_inps.phase,
            "task_idxs": move_inps.task_idxs,
            "valid_actions": move_inps.valid_actions,
            "h0": h,
            "c0": c,
        }
        with torch.no_grad():
            log_probs, value, _ = model(inps)
            log_probs = log_probs.numpy()
            value = value.numpy()

        ort_log_probs, ort_value, h, c = ort_model.run(None, ort_inps)

        assert np.isclose(log_probs, ort_log_probs, atol=1e-5).all()
        assert np.isclose(value, ort_value, atol=1e-5).all()
        action_idxs = np.argmax(log_probs, axis=1)
        batch_rollout.move(action_idxs, log_probs)

        i += 1

    click.echo("✓ Validation passed")


@click.command()
@click.argument("model-path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-name",
    "-o",
    default="model.onnx",
    help="Output ONNX model filename (default: model.onnx)",
)
@click.option(
    "--batch-size", default=5, help="Batch size for sample inputs (default: 5)"
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate ONNX export against PyTorch model (default: True)",
)
def export_onnx(model_path: Path, output_name: str, batch_size: int, validate: bool):
    """
    Export a trained Crew AI model to ONNX format.

    MODEL_PATH should point to a directory containing 'settings.pth' and 'checkpoint.pth' files.

    Example:
        python export_onnx.py /path/to/model/run_16 --output-name my_model.onnx
    """
    click.echo(f"Loading model from: {model_path}")

    pv_model, settings, hp = load_model_for_eval(model_path, onnx=True)

    # Generate sample inputs using the game engine
    click.echo(f"Generating sample inputs with batch size {batch_size}...")
    cpp_settings = settings.to_cpp()
    batch_rollout = cpp_game.BatchRollout(cpp_settings, batch_size)
    engine_seeds = list(range(batch_size))
    batch_rollout.reset_state(engine_seeds)
    move_inps = batch_rollout.get_move_inputs()

    # Initialize LSTM hidden states
    h = c = torch.zeros((hp.hist_num_layers, batch_size, hp.hist_hidden_dim))

    # Prepare input tensors
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
        torch.from_numpy(move_inps.valid_actions),
        h,
        c,
    )

    # Define input/output names and dynamic axes
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
        "valid_actions",
        "h0",
        "c0",
    ]
    output_names = ["log_probs", "value", "h1", "c1"]

    # Define which dimension is the batch dimension for each tensor
    inp_batch_idxs = [0] * 12 + [
        1
    ] * 2  # First 12 inputs have batch dim 0, last 2 have batch dim 1
    out_batch_idxs = [
        0,
        0,
        1,
        1,
    ]  # log_probs, value have batch dim 0; h1, c1 have batch dim 1

    assert len(inp_batch_idxs) == len(inps) == len(input_names)
    assert len(out_batch_idxs) == len(output_names)

    # Set up dynamic axes for variable batch size
    dynamic_axes = {}
    for name, batch_idx in zip(input_names, inp_batch_idxs):
        dynamic_axes[name] = {batch_idx: "B"}
    for name, batch_idx in zip(output_names, out_batch_idxs):
        dynamic_axes[name] = {batch_idx: "B"}

    click.echo("Dynamic axes configuration:")
    for name, axes in dynamic_axes.items():
        click.echo(f"  {name}: {axes}")

    # Test forward pass
    click.echo("Testing forward pass...")
    with torch.no_grad():
        outs = pv_model(*inps)

    click.echo("Output shapes:")
    for i, out in enumerate(outs):
        click.echo(f"  {output_names[i]}: {out.shape}")

    # Export to ONNX
    click.echo(f"Exporting to ONNX: {output_name}")
    with torch.no_grad():
        torch.onnx.export(
            pv_model,
            inps,
            str(model_path / output_name),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    click.echo("Checking ONNX model...")
    onnx_model = onnx.load(str(model_path / output_name))
    onnx.checker.check_model(onnx_model)

    click.echo("ONNX model inputs:")
    for t in onnx_model.graph.input:
        dims = [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
        click.echo(f"  {t.name}: {dims}")

    if validate:
        validate_model(model_path)

    click.echo(f"✓ Successfully exported model to {output_name}")


if __name__ == "__main__":
    export_onnx()
