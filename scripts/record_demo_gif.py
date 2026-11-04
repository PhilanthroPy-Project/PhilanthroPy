"""Record the README's CLI demo as an animated GIF, without vhs or asciinema.

Builds a synthetic CiviCRM-shaped gift export from
:func:`philanthropy.datasets.make_donor_panel`, runs the real ``philanthropy``
commands on it in a temporary directory, and renders what they actually print
to ``docs/assets/cli_demo.gif``: each command is "typed" a few characters per
frame, then its captured output appears. Nothing on screen is written by hand,
so the GIF cannot drift from what the CLI says; re-run this after a change to
the CLI's output.

Needs Pillow, which the ``dev`` extra already installs through matplotlib. No
network.

Usage:
    python scripts/record_demo_gif.py
    python scripts/record_demo_gif.py --out /tmp/demo.gif
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = [
    "philanthropy features --source civicrm --data gifts.csv --out features.csv",
    "philanthropy check --task lapse --source civicrm --data gifts.csv",
]
WIDTH, PAD, LINE_H, FONT_SIZE = 1000, 18, 22, 15
BG, FG, PROMPT, DIM = (24, 26, 33), (220, 223, 228), (120, 200, 140), (140, 145, 155)
CHARS_PER_FRAME, TYPE_MS, OUTPUT_MS, END_MS = 4, 45, 1600, 4000


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Menlo.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, FONT_SIZE)
    return ImageFont.load_default(size=FONT_SIZE)


def _write_gifts(workdir: Path) -> None:
    from philanthropy.datasets import make_donor_panel

    gifts = make_donor_panel(n_donors=3000, random_state=0)["gifts"]
    export = gifts.rename(columns={
        "donor_id": "Contact ID", "gift_date": "Date Received",
        "gift_amount": "Total Amount", "appeal": "Financial Type",
    })
    export["Contribution Status"] = "Completed"
    export[["Contact ID", "Date Received", "Total Amount", "Financial Type",
            "Contribution Status"]].to_csv(workdir / "gifts.csv", index=False)


def _run(command: str, workdir: Path) -> list[str]:
    # Same entry point as the console script, without needing it on PATH.
    args = [sys.executable, "-m", "philanthropy.cli", *command.split()[1:]]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    done = subprocess.run(args, cwd=workdir, env=env, capture_output=True,
                          text=True, check=True)
    return (done.stdout + done.stderr).rstrip("\n").splitlines()


def _frame(lines: list[tuple[str, tuple[int, int, int]]], height: int,
           font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Image.Image:
    img = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)
    for i, (text, colour) in enumerate(lines):
        draw.text((PAD, PAD + i * LINE_H), text, font=font, fill=colour)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "docs/assets/cli_demo.gif"))
    out = Path(parser.parse_args().out)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        _write_gifts(workdir)
        outputs = [_run(cmd, workdir) for cmd in COMMANDS]

    header = ("# synthetic CiviCRM gift export from make_donor_panel", DIM)
    n_lines = 1 + sum(1 + len(o) for o in outputs) + 1
    height = 2 * PAD + n_lines * LINE_H
    font = _font()

    frames: list[Image.Image] = []
    durations: list[int] = []
    shown: list[tuple[str, tuple[int, int, int]]] = [header]
    for cmd, output in zip(COMMANDS, outputs):
        for end in range(0, len(cmd) + 1, CHARS_PER_FRAME):
            frames.append(_frame(shown + [("$ " + cmd[:end], PROMPT)], height, font))
            durations.append(TYPE_MS)
        shown = shown + [("$ " + cmd, PROMPT)] + [(line, FG) for line in output]
        frames.append(_frame(shown, height, font))
        durations.append(OUTPUT_MS)
    frames.append(_frame(shown + [("$", PROMPT)], height, font))
    durations.append(END_MS)

    # A shared palette keeps the file small; the image has only a few colours.
    palette = [f.quantize(colors=16, method=Image.Quantize.MEDIANCUT) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    palette[0].save(out, save_all=True, append_images=palette[1:],
                    duration=durations, loop=0, optimize=True)
    print(f"Wrote {len(frames)} frames to {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
