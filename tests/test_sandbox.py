import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sandbox import execute_code, static_check


def test_blocks_import_os():
    assert static_check("import os\nprint(os.getcwd())") == "os"


def test_blocks_import_sys():
    assert static_check("import sys\nprint(sys.argv)") == "sys"


def test_blocks_subprocess():
    assert static_check("import subprocess\nsubprocess.run(['ls'])") == "subprocess"


def test_blocks_requests():
    assert static_check("import requests\nrequests.get('http://x')") == "requests"


def test_blocks_urllib():
    assert static_check("import urllib.request") == "urllib"


def test_blocks_socket():
    assert static_check("import socket\nsocket.socket()") == "socket"


def test_blocks_shutil():
    assert static_check("import shutil\nshutil.rmtree('/')") == "shutil"


def test_blocks_eval():
    assert static_check("eval('1+1')") == "eval("


def test_blocks_exec():
    assert static_check("exec('print(1)')") == "exec("


def test_blocks_dunder_import():
    assert static_check("__import__('os')") == "__import__"


def test_blocks_open_write_outside_sandbox():
    violation = static_check("open('/etc/passwd', 'w').write('x')")
    assert violation is not None
    assert "open()" in violation


def test_allows_open_write_relative_path():
    assert static_check("open('chart.png', 'wb').write(b'')") is None


def test_allows_whitelisted_analysis_code():
    code = (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "df = pd.DataFrame({'a': [1, 2, 3]})\n"
        "print(df.describe())\n"
    )
    assert static_check(code) is None


def test_execute_code_happy_path(tmp_path):
    chart_path = tmp_path / "chart.png"
    code = "print('hello world')\n"
    result = execute_code(code, tmp_path, chart_path)
    assert result.success
    assert "hello world" in result.stdout
    assert result.stderr == ""
    assert result.chart_created is False


def test_execute_code_creates_chart(tmp_path):
    chart_path = tmp_path / "chart.png"
    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        f"plt.plot([1, 2, 3])\n"
        f"plt.savefig(r'{chart_path}')\n"
        "print('chart saved')\n"
    )
    result = execute_code(code, tmp_path, chart_path)
    assert result.success
    assert result.chart_created


def test_execute_code_captures_traceback(tmp_path):
    chart_path = tmp_path / "chart.png"
    code = "raise KeyError('missing_column')\n"
    result = execute_code(code, tmp_path, chart_path)
    assert not result.success
    assert "KeyError" in result.stderr


def test_execute_code_empty_stdout_is_failure(tmp_path):
    chart_path = tmp_path / "chart.png"
    code = "x = 1 + 1\n"
    result = execute_code(code, tmp_path, chart_path)
    assert not result.success


def test_execute_code_blocked_import_short_circuits(tmp_path):
    chart_path = tmp_path / "chart.png"
    code = "import os\nprint(os.getcwd())\n"
    result = execute_code(code, tmp_path, chart_path)
    assert not result.success
    assert result.stderr == "blocked import: os"
