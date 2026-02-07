import subprocess
import sys
from pathlib import Path

def test_test_py_prints_cali_pred():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "src" / "nsbi_common_utils" / "test.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    assert "Testing cali_pred function..." in output