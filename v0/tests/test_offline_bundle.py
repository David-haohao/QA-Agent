"""Contract tests for the Windows offline installation bundle."""

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = PROJECT_ROOT / "download"


class OfflineBundleScriptTests(unittest.TestCase):
    def test_installer_is_network_free_and_safe_for_existing_venv(self):
        installer = BUNDLE_ROOT / "install_offline.ps1"
        self.assertTrue(installer.is_file(), "offline installer must be bundled")
        text = installer.read_text(encoding="utf-8")
        self.assertIn("--no-index", text)
        self.assertIn("--find-links", text)
        self.assertIn("Test-Path -LiteralPath $venvPython", text)
        self.assertIn("-m pip check", text)

    def test_verifier_checks_manifest_without_network(self):
        verifier = BUNDLE_ROOT / "verify_offline.ps1"
        self.assertTrue(verifier.is_file(), "offline verifier must be bundled")
        text = verifier.read_text(encoding="utf-8")
        self.assertIn("Get-FileHash", text)
        self.assertIn("SHA256SUMS.txt", text)
        self.assertNotIn("Invoke-WebRequest", text)


if __name__ == "__main__":
    unittest.main()
