import json
import tempfile
import unittest
from pathlib import Path

import installer


class InstallerTests(unittest.TestCase):
    def test_default_model_uses_claude_alias(self):
        self.assertEqual(installer.DEFAULT_MODEL, "claude-code-sonnet")

    def test_configure_merges_files_without_exposing_unrelated_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            claude = root / ".claude" / "settings.json"
            ide = root / "Antigravity" / "User" / "settings.json"
            claude.parent.mkdir(parents=True)
            ide.parent.mkdir(parents=True)
            claude.write_text('{"env":{"KEEP":"yes"},"theme":"dark"}', encoding="utf-8")
            ide.write_text('{"editor.fontSize":14}', encoding="utf-8")

            installer.configure(
                "secret-token",
                "https://example.test/",
                "hosted_vllm/test-model",
                claude_settings=claude,
                ide_settings=[ide],
                should_install_extension=False,
            )

            claude_data = json.loads(claude.read_text(encoding="utf-8"))
            ide_data = json.loads(ide.read_text(encoding="utf-8"))
            self.assertEqual(claude_data["theme"], "dark")
            self.assertEqual(claude_data["env"]["KEEP"], "yes")
            self.assertEqual(claude_data["env"]["ANTHROPIC_AUTH_TOKEN"], "secret-token")
            self.assertEqual(claude_data["env"]["ANTHROPIC_BASE_URL"], "https://example.test")
            self.assertTrue(ide_data["claudeCode.disableLoginPrompt"])
            self.assertEqual(ide_data["editor.fontSize"], 14)
            self.assertTrue(list(claude.parent.glob("settings.json.backup-*")))
            self.assertTrue(list(ide.parent.glob("settings.json.backup-*")))

    def test_invalid_url_is_rejected(self):
        with self.assertRaises(ValueError):
            installer.validate_base_url("example.test")

    def test_jsonc_comments_and_trailing_commas_are_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = Path(temp) / "settings.json"
            settings.write_text(
                '{\n  // editor preference\n  "editor.fontSize": 14,\n}\n',
                encoding="utf-8",
            )
            installer.merge_ide_settings(
                settings, "secret-token", "https://example.test", "model"
            )
            data = json.loads(settings.read_text(encoding="utf-8"))
            self.assertEqual(data["editor.fontSize"], 14)
            self.assertTrue(data["claudeCode.disableLoginPrompt"])

    def test_empty_settings_file_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = Path(temp) / "settings.json"
            settings.write_text("", encoding="utf-8")
            installer.merge_claude_settings(
                settings, "secret-token", "https://example.test", "model"
            )
            data = json.loads(settings.read_text(encoding="utf-8"))
            self.assertEqual(data["env"]["ANTHROPIC_MODEL"], "model")

    def test_dry_run_does_not_write_files(self):
        with tempfile.TemporaryDirectory() as temp:
            claude = Path(temp) / "settings.json"
            installer.configure(
                "secret-token",
                "https://example.test",
                "model",
                claude_settings=claude,
                ide_settings=[],
                should_install_extension=False,
                dry_run=True,
            )
            self.assertFalse(claude.exists())


if __name__ == "__main__":
    unittest.main()
