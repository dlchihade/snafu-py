import pathlib
import py_compile
import pytest


def _collect_script_paths() -> list[pathlib.Path]:
	base = pathlib.Path(__file__).resolve().parents[1] / "output" / "scripts"
	paths: list[pathlib.Path] = []
	# Top-level scripts (e.g., archetype symlink)
	paths.extend(base.glob("*.py"))
	# Categorized scripts in subfolders
	paths.extend(base.glob("*/*.py"))
	# Deduplicate and sort
	unique = sorted({p.resolve() for p in paths})
	return unique


SCRIPTS = _collect_script_paths()


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(path: pathlib.Path) -> None:
	"""Ensure every script at least compiles (catches syntax errors)."""
	py_compile.compile(str(path), doraise=True)


