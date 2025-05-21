import selectors
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import click
from loguru import logger


def run_cmds(cmds):
    selector = selectors.DefaultSelector()
    processes = []

    def register_stream(stream, tag, is_stdout):
        selector.register(stream, selectors.EVENT_READ, data=(tag, is_stdout))

    # Launch subprocesses and register their stdout/stderr
    for seed, cmd in cmds.items():
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,  # text mode
        )
        processes.append((seed, cmd, proc))
        register_stream(proc.stdout, f"[stdout-{seed}]", True)
        register_stream(proc.stderr, f"[stderr-{seed}]", False)

    # Stream outputs
    alive_procs = set(proc for _, _, proc in processes)
    while alive_procs:
        for key, _ in selector.select(timeout=0.1):
            tag, is_stdout = key.data
            line = key.fileobj.readline()
            if line:
                (sys.stdout if is_stdout else sys.stderr).write(f"{tag}: {line}")
            else:
                selector.unregister(key.fileobj)
                key.fileobj.close()

        # Check for exit and errors
        for seed, cmd, proc in processes:
            if proc in alive_procs and proc.poll() is not None:
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Subprocess {seed} failed with code {proc.returncode}"
                    )
                alive_procs.remove(proc)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option(
    "--outdir",
    type=Path,
    help="Outdir",
    required=True,
)
@click.option(
    "--num-seeds",
    type=int,
    help="Number of seeds",
    required=True,
)
@click.option(
    "--clean",
    is_flag=True,
    help="Clean outdir",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Must set --resume to run on existing outdir.",
)
@click.option(
    "--autoindex-runs",
    is_flag=True,
    help="Auto-index new runs.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def main(
    outdir: Path,
    num_seeds: int,
    clean: bool,
    resume: bool,
    autoindex_runs: bool,
    args: list[str],
):
    outdir = outdir.resolve()
    assert autoindex_runs == ("run" not in outdir.name), (
        "We expect outdir to be named with run_* unless autoindex_runs=True"
    )

    if autoindex_runs:
        max_run_idx = max(
            [int(x.name.split("_")[-1]) for x in outdir.glob("run_*")],
            default=-1,
        )
        if clean or resume:
            outdir = outdir / f"run_{max(max_run_idx, 0)}"
        else:
            outdir = outdir / f"run_{max_run_idx + 1}"

    if outdir.exists():
        if clean:
            logger.info(f"** Cleaning outdir {outdir} **")
            shutil.rmtree(outdir)
        elif not resume:
            raise Exception("Must set --clean or --resume to run on existing outdir.")

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "_keep").touch()

    cmds = {}
    for seed in range(num_seeds):
        seed_dir = outdir / f"seed_{seed}"

        cmd = [
            sys.executable,
            "-m",
            "src.ai.train",
            "--outdir",
            str(seed_dir),
            "--no-error-catch",
            "--seed",
            str(seed + 1337),
        ]
        if resume:
            cmd.append("--resume")
        cmd += list(args)
        cmds[seed] = cmd

    for seed, cmd in cmds.items():
        logger.info(f"Launching seed={seed}: {shlex.join(cmd)}")

    run_cmds(cmds)


if __name__ == "__main__":
    main()
