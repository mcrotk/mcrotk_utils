"""Draw the images in the subpackages' READMEs, from the code shown in them.

A README example that draws a plot is a fenced ``python`` block followed (after any blank lines) by an image link to
a file in the README's ``images`` folder, e.g. ``![Week over week](images/week_over_week.png)``. This script runs
each such block and saves the figure it drew as that image, so the code shown is the code that drew the image.

The blocks of one README run in order, in one namespace, so later blocks can use earlier blocks' imports. Each
block should draw one figure; the figure current at the end of the block is the one saved.

Usage: ``python scripts/make_readme_images.py [--output-dir DIR]``. By default each image is written to its README's
``images`` folder; with ``--output-dir``, all of them go to that folder instead (as the tests do, to check that
the examples run without changing the repository).
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Drawn without a display: chosen before pyplot is imported
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "mcrotk_utils"

# A python block, then (after blank lines) an image link into the README's images folder
# (the code can't contain a fence, so a block without an image never runs on into the next block)
_IMAGE_EXAMPLE = re.compile(
    r"^```python\n(?P<code>(?:(?!^```)[\s\S])*?)^```\n\s*!\[[^\]]*\]\(images/(?P<name>[^)\s]+\.png)\)", re.M
)

# Resolution of the saved images: enough to be sharp on most screens while keeping the files small
DPI = 110


@dataclass(frozen=True)
class ImageExample:
    """One example that draws an image: its README, its code, and the image's file name."""

    readme: Path
    code: str
    image_name: str


def readmes():
    """Return the paths of the subpackages' READMEs."""
    return sorted(PACKAGE_DIR.glob("*/README.md"))


def image_examples(readme):
    """Return the examples in ``readme`` that draw an image, in order."""
    text = readme.read_text()
    return [ImageExample(readme, match["code"], match["name"]) for match in _IMAGE_EXAMPLE.finditer(text)]


def draw(example, namespace):
    """Run ``example``'s code in ``namespace``, and return the figure it drew."""
    plt.close("all")
    exec(compile(example.code, f"{example.readme} ({example.image_name})", "exec"), namespace)
    return plt.gcf()


def make_images(output_dir=None):
    """Draw every README image, writing it to ``output_dir`` or its README's images folder; return the paths."""
    written = []
    for readme in readmes():
        namespace = {"__name__": "__readme__"}
        for example in image_examples(readme):
            figure = draw(example, namespace)
            folder = Path(output_dir) if output_dir is not None else readme.parent / "images"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / example.image_name
            # metadata without the matplotlib version, so redrawing with another version changes only the pixels
            figure.savefig(path, dpi=DPI, bbox_inches="tight", metadata={"Software": None})
            plt.close(figure)
            written.append(path)
    return written


def main():
    """Parse the command line and draw the images."""
    parser = argparse.ArgumentParser(description="Draw the images in the subpackages' READMEs.")
    parser.add_argument("--output-dir", help="write all the images here instead of the READMEs' images folders")
    arguments = parser.parse_args()
    for path in make_images(arguments.output_dir):
        print(path)


if __name__ == "__main__":
    main()
