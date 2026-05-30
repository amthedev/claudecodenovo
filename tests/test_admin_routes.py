import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxy_app.admin_routes import _root_test_key


def test_original_root_key_can_be_tested():
    assert _root_test_key("env-root", False, None) == "env-root"


def test_rotated_root_key_uses_current_reveal():
    reveal = {"key_id": "root", "key_value": "rotated-root"}
    assert _root_test_key("old-env-root", True, reveal) == "rotated-root"


def test_rotated_root_key_without_reveal_does_not_use_old_env_key():
    assert _root_test_key("old-env-root", True, None) == ""
