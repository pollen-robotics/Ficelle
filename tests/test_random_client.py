import subprocess
import sys
from pathlib import Path


def test_random_client_against_fake_server(server):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "examples" / "random_client.py"),
         "--host", "127.0.0.1", "--port", str(server.port), "-n", "3"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "rtt p50" in result.stdout
    assert "actions shape OK" in result.stdout
