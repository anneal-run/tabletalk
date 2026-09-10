"""The sandbox refuses the obvious escapes and kills runaway scripts."""

import pytest

from tabletalk.tools.sandbox import SandboxError, check_code, run_python


@pytest.mark.parametrize(
    "code",
    [
        "import os; os.system('echo hi')",
        "from subprocess import run",
        "import socket",
        "open('/etc/passwd').read()",
        "__import__('os')",
        "().__class__.__bases__",
        "import pandas as pd; pd.read_csv('https://example.com/x.csv')",
        "eval('1+1')",
    ],
)
def test_dangerous_code_is_refused_before_running(code):
    with pytest.raises(SandboxError):
        check_code(code)


def test_allowed_code_runs_and_reads_result_csv():
    res = run_python(
        "import pandas as pd\ndf = pd.read_csv('result.csv')\nprint(df['x'].sum())",
        result_csv="x\n1\n2\n3\n",
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "6"


def test_chart_is_copied_out(tmp_path):
    out = tmp_path / "c.png"
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.bar(['a', 'b'], [1, 2])\n"
        "plt.savefig('chart.png')\n"
    )
    res = run_python(code, result_csv=None, chart_out=str(out))
    assert res.returncode == 0, res.stderr
    assert res.chart_path == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_infinite_loop_is_killed():
    res = run_python("while True:\n    pass\n", result_csv=None, timeout_s=1.0)
    assert res.timed_out is True
    assert "killed" in res.stderr
