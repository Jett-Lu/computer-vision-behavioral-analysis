import os
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_policy import RetentionRule, apply_retention, plan_retention, ensure_private_directory
from data_policy import _safe_path
import data_policy


class DataPolicyTest(unittest.TestCase):
    def test_system_alias_exception_is_narrow_and_checks_ownership(self):
        # Model the OS-owned alias without requiring a macOS host or root access.
        alias = MagicMock(spec=Path)
        alias.__str__.return_value = "/var"
        alias.readlink.return_value = Path("private/var")
        alias.lstat.return_value.st_uid = 0
        alias.parent.lstat.return_value.st_uid = 0
        alias.parent.lstat.return_value.st_mode = 0o40755
        with patch("sys.platform", "darwin"):
            self.assertEqual(data_policy._macos_system_alias_target(alias), Path("/private/var"))
            alias.readlink.return_value = Path("/outside")
            self.assertIsNone(data_policy._macos_system_alias_target(alias))
            alias.readlink.return_value = Path("private/var")
            alias.lstat.return_value.st_uid = 501
            self.assertIsNone(data_policy._macos_system_alias_target(alias))
            alias.lstat.return_value.st_uid = 0
            alias.parent.lstat.return_value.st_mode = 0o40777
            self.assertIsNone(data_policy._macos_system_alias_target(alias))
            alias.parent.lstat.return_value.st_mode = 0o40755
            alias.__str__.return_value = "/home/user/var"
            self.assertIsNone(data_policy._macos_system_alias_target(alias))
        alias.__str__.return_value = "/var"
        with patch("sys.platform", "linux"):
            self.assertIsNone(data_policy._macos_system_alias_target(alias))

    @unittest.skipUnless(sys.platform == "darwin", "macOS system aliases")
    def test_macos_system_alias_allows_private_creation_and_retention(self):
        for alias in (Path("/var"), Path("/tmp")):
            with self.subTest(alias=alias):
                base = alias / "tmp" if alias.name == "var" else alias
                with tempfile.TemporaryDirectory(dir=base) as directory:
                    root = Path(directory) / "private"
                    ensure_private_directory(root)
                    self.assertEqual(root.stat().st_mode & 0o777, 0o700)
                    image = root / "portrait.jpg"
                    image.write_bytes(b"expired")
                    os.utime(image, (1, 1))
                    plan = plan_retention([RetentionRule("identities", root, 1)])
                    self.assertEqual(plan[0].path, image.resolve())
                    self.assertEqual(apply_retention(plan), 1)
                    self.assertFalse(image.exists())

    def test_linked_ancestor_and_dangling_link_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root / "outside"
            outside.mkdir()
            link = root / "link"
            dangling = root / "dangling"
            try:
                link.symlink_to(outside, target_is_directory=True)
                dangling.symlink_to(root / "missing", target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation requires OS permission")
            for target in (link / "new", dangling):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    ensure_private_directory(target)
            self.assertEqual(list(outside.iterdir()), [])

    def test_hardlinked_retention_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original = root / "original.jpg"
            original.write_bytes(b"keep")
            linked = root / "linked.jpg"
            os.link(original, linked)
            os.utime(original, (1, 1))
            with self.assertRaises(ValueError):
                plan_retention([RetentionRule("identities", root, 1)])
            self.assertEqual(original.read_bytes(), b"keep")
            self.assertEqual(linked.read_bytes(), b"keep")

    @unittest.skipUnless(sys.platform == "darwin", "macOS system aliases")
    def test_macos_alias_does_not_allow_user_links_or_alias_roots(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            marker = outside / "portrait.jpg"
            marker.write_bytes(b"keep")
            link = root / "link"
            link.symlink_to(outside, target_is_directory=True)
            for target in (link, link / "child", Path("/var"), Path("/tmp")):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    _safe_path(target)
            self.assertEqual(marker.read_bytes(), b"keep")

    @unittest.skipUnless(os.name == "nt", "Windows junction verification")
    def test_windows_junction_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            marker = outside / "portrait.jpg"
            marker.write_bytes(b"keep")
            link = root / "junction"
            subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(outside)],
                check=True,
                capture_output=True,
            )
            try:
                with self.assertRaises(ValueError):
                    plan_retention([RetentionRule("identities", link, 30)])
                self.assertEqual(marker.read_bytes(), b"keep")
            finally:
                link.rmdir()

    def test_permission_failure_does_not_leave_unprotected_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "private"
            with patch("data_policy._restrict_access", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    ensure_private_directory(target)
            self.assertFalse(target.exists())

    @unittest.skipUnless(os.name == "nt", "Windows ACL verification")
    def test_windows_acl_has_only_current_account_and_system(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "private"
            ensure_private_directory(target)
            env = os.environ.copy()
            env["MONITOR_ACL_TARGET"] = str(target)
            result = subprocess.check_output(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "$acl = (Get-Item -LiteralPath $env:MONITOR_ACL_TARGET).GetAccessControl(); "
                    "@{ Protected = $acl.AreAccessRulesProtected; "
                    "Owner = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
                    "Sids = @($acl.Access | ForEach-Object { $_.IdentityReference.Translate("
                    "[System.Security.Principal.SecurityIdentifier]).Value }) } | ConvertTo-Json",
                ],
                env=env,
                text=True,
            )
            acl = json.loads(result)
            self.assertTrue(acl["Protected"])
            self.assertEqual(set(acl["Sids"]), {acl["Owner"], "S-1-5-18"})

    def test_preview_selects_only_expired_managed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("old_annotated.mp4", "fresh_annotated.mp4", "original.mp4"):
                (root / name).write_bytes(b"media")
            os.utime(root / "old_annotated.mp4", (1, 1))
            os.utime(root / "original.mp4", (1, 1))
            rule = RetentionRule("recordings", root, 30)
            plan = plan_retention([rule])
            self.assertEqual([item.path.name for item in plan], ["old_annotated.mp4"])
            self.assertTrue(plan[0].path.exists())
            apply_retention(plan)
            self.assertFalse((root / "old_annotated.mp4").exists())
            self.assertTrue((root / "original.mp4").exists())

    def test_changed_file_cannot_be_deleted_using_stale_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "portrait.jpg"
            path.write_bytes(b"old")
            os.utime(path, (1, 1))
            plan = plan_retention([RetentionRule("identities", root, 365)])
            path.write_bytes(b"new enrollment")
            with self.assertRaises(ValueError):
                apply_retention(plan)
            self.assertTrue(path.exists())

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            link = root / "link"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation requires OS permission")
            with self.assertRaises(ValueError):
                plan_retention([RetentionRule("identities", link, 30)])

    def test_zero_retention_is_rejected(self):
        with self.assertRaises(ValueError):
            RetentionRule("identities", Path("enrollments"), 0)

    def test_new_private_directory_and_existing_directory_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "private"
            ensure_private_directory(target)
            self.assertTrue(target.is_dir())
            if os.name != "nt":
                self.assertEqual(target.stat().st_mode & 0o777, 0o700)
