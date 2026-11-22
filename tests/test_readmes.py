"""Tests for the READMEs: the package's, and each subpackage's.

Each subpackage's README must list every public name, and the package's README must link to each one. The
examples in every README (fenced ``pycon`` blocks, in doctest format) are run, and must give the output shown. The
examples that draw images (``python`` blocks followed by an image link) are run by ``scripts/make_readme_images.py``,
which is tested here: it must draw every image, and every image a README shows must exist.
"""

import doctest
import importlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src" / "mcrotk_utils"
PACKAGE_README = ROOT / "README.md"
IMAGES_SCRIPT = ROOT / "scripts" / "make_readme_images.py"

# Every subpackage: the folders under the package with an __init__.py
SUBPACKAGES = sorted(path.parent.name for path in PACKAGE_DIR.glob("*/__init__.py"))

# A fenced block of examples: ```pycon, its lines, and the closing ```
_PYCON_BLOCK = re.compile(r"^```pycon\n(.*?)^```$", re.MULTILINE | re.DOTALL)

# An image in Markdown, ![description](path), capturing the path
_IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def readme_of(subpackage):
    """Return the path of a subpackage's README."""
    return PACKAGE_DIR / subpackage / "README.md"


def examples_text(markdown):
    """Return the lines of ``markdown``'s ``pycon`` blocks, with every other line blanked.

    Blanking the other lines (rather than dropping them) keeps each example on its line of the file, so doctest's
    failure reports point at the right line; the blank lines also end each example's expected output.
    """
    # The numbers (from 0) of the lines inside the blocks
    inside: set[int] = set()
    for match in _PYCON_BLOCK.finditer(markdown):
        first_line = markdown.count("\n", 0, match.start(1))
        inside.update(range(first_line, first_line + match.group(1).count("\n")))
    return "\n".join(line if number in inside else "" for number, line in enumerate(markdown.split("\n")))


def run_examples(path):
    """Run the examples in the README at ``path`` in a fresh namespace.

    Returns the number of examples that failed, the number run, and the report of the failures.
    """
    test = doctest.DocTestParser().get_doctest(examples_text(path.read_text()), {}, path.name, str(path), 0)
    report = io.StringIO()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    runner.run(test, out=report.write)
    return runner.failures, runner.tries, report.getvalue()


def images_script():
    """Import ``scripts/make_readme_images.py`` (which isn't in a package) and return it as a module.

    Importing it selects matplotlib's Agg backend, so this is only called by the tests that draw.
    """
    spec = importlib.util.spec_from_file_location("make_readme_images", IMAGES_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_subpackages_are_found():
    """The list of subpackages isn't empty, so the tests below over it test something."""
    assert "misc_utils" in SUBPACKAGES and len(SUBPACKAGES) >= 7


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_every_subpackage_has_a_readme(subpackage):
    """Each subpackage's folder has a README.md."""
    assert readme_of(subpackage).is_file()


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_readme_lists_every_public_name(subpackage):
    """Each name in a subpackage's ``__all__`` appears in its README, as code (in backticks)."""
    module = importlib.import_module(f"mcrotk_utils.{subpackage}")
    text = readme_of(subpackage).read_text()
    missing = [name for name in module.__all__ if f"`{name}`" not in text]
    assert not missing, f"{subpackage}'s README doesn't mention {missing}"


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_package_readme_links_to_each_subpackage_readme(subpackage):
    """The package's README links to each subpackage's README, by its path in the repository."""
    assert f"(src/mcrotk_utils/{subpackage}/README.md)" in PACKAGE_README.read_text()


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_subpackage_readme_links_back_to_the_package_readme(subpackage):
    """Each subpackage's README links back to the package's README, by a relative path that leads there."""
    text = readme_of(subpackage).read_text()
    links = re.findall(r"\]\(([^)]*README\.md)\)", text)
    assert any((readme_of(subpackage).parent / link).resolve() == PACKAGE_README for link in links)


@pytest.mark.parametrize(
    "path", [PACKAGE_README] + [readme_of(name) for name in SUBPACKAGES], ids=lambda p: p.parent.name
)
def test_readme_examples_give_the_output_shown(path):
    """Every example in the README runs and gives the output it shows."""
    # plots in the examples are drawn without a display (matplotlib is imported here, not at the top, so that the
    # other tests in this file don't need it)
    import matplotlib

    matplotlib.use("Agg")
    failures, tries, report = run_examples(path)
    assert failures == 0, report
    # every subpackage's README has examples, so none running would mean they weren't found
    assert tries > 0 or path == PACKAGE_README


def test_examples_text_keeps_the_blocks_on_their_lines():
    """``examples_text`` keeps the lines of ``pycon`` blocks where they are, and blanks the rest."""
    markdown = "# Title\n\n```pycon\n>>> 1 + 1\n2\n```\n\n```python\nx = 1\n```\n"
    assert examples_text(markdown).split("\n") == ["", "", "", ">>> 1 + 1", "2", "", "", "", "", "", ""]


def test_a_wrong_output_is_reported(tmp_path):
    """An example whose output differs from what's shown counts as a failure, so the checks above can fail."""
    path = tmp_path / "README.md"
    path.write_text("```pycon\n>>> 1 + 1\n3\n```\n")
    failures, tries, report = run_examples(path)
    assert (failures, tries) == (1, 1) and "Expected:" in report


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_every_image_shown_exists(subpackage):
    """Every image a subpackage's README shows is a file in the repository, by its path relative to the README."""
    readme = readme_of(subpackage)
    missing = [link for link in _IMAGE_LINK.findall(readme.read_text()) if not (readme.parent / link).is_file()]
    assert not missing, f"{subpackage}'s README shows images that don't exist: {missing}"


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_every_image_shown_is_drawn_by_an_example(subpackage):
    """Every image a README shows follows an example the script finds, so none is left out of redrawing."""
    readme = readme_of(subpackage)
    shown = _IMAGE_LINK.findall(readme.read_text())
    drawn = [f"images/{example.image_name}" for example in images_script().image_examples(readme)]
    assert drawn == shown


def test_the_image_examples_draw_their_images(tmp_path):
    """The script runs every image example and writes each image, as a PNG, to the folder given."""
    script = images_script()
    expected = sorted(example.image_name for readme in script.readmes() for example in script.image_examples(readme))
    written = script.make_images(tmp_path)
    # some examples exist (the plotting README's), so this tests something
    assert expected
    assert sorted(path.name for path in written) == expected
    assert sorted(path.name for path in tmp_path.iterdir()) == expected
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in written)


def test_a_block_without_an_image_is_not_an_image_example(tmp_path):
    """A ``python`` block not followed by an image is skipped, not run together with a later block that is."""
    readme = tmp_path / "README.md"
    readme.write_text("```python\nnot drawn\n```\n\nText.\n\n```python\nx = 1\n```\n\n![A plot](images/plot.png)\n")
    examples = images_script().image_examples(readme)
    assert [(example.code, example.image_name) for example in examples] == [("x = 1\n", "plot.png")]
