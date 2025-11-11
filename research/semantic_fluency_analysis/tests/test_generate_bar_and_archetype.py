import subprocess
import sys
from pathlib import Path


def _python_exe() -> str:
	return sys.executable or "python3"


def test_create_exploit_explore_bar_runs(tmp_path: Path) -> None:
	root = Path(__file__).resolve().parents[1]
	metrics = root / "output" / "NATURE_REAL_metrics.csv"
	script = root / "create_exploit_explore_bar.py"
	assert metrics.exists(), f"Missing metrics at {metrics}"
	assert script.exists(), f"Missing script at {script}"
	# Run the script; it writes to output folder by default
	cmd = [_python_exe(), str(script)]
	res = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
	assert res.returncode == 0, res.stderr or res.stdout
	# Verify outputs exist
	out_dir = root / "output"
	assert (out_dir / "exploit_explore_bar.png").exists()
	assert (out_dir / "exploit_explore_bar.pdf").exists()


def test_archetype_analysis_runs(tmp_path: Path) -> None:
	repo_root = Path(__file__).resolve().parents[3]  # snafu-py
	arch = repo_root / "research" / "semantic_fluency_analysis" / "output" / "archetype" / "archetype_analysis.py"
	assert arch.exists(), f"Missing archetype at {arch}"
	# Use out-tag temp
	out_tag = "pytest_demo"
	cmd = [_python_exe(), str(arch), "--out-tag", out_tag]
	res = subprocess.run(cmd, cwd=str(arch.parent), capture_output=True, text=True)
	assert res.returncode == 0, res.stderr or res.stdout
	fig_dir = arch.parent.parent / "figures" / out_tag
	assert any(fig_dir.glob("*.png")), "No PNG produced by archetype"
	assert any(fig_dir.glob("*.pdf")), "No PDF produced by archetype"


