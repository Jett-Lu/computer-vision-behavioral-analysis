"""Local-account access controls and explicit, previewable data retention.

Run maintenance with monitoring stopped. These filesystem controls are not
encryption, secure erasure, or protection against the account owner/admin.
"""

from dataclasses import dataclass
from contextlib import contextmanager
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import tempfile
from typing import Iterator
from io import FileIO, TextIOWrapper

from config import AppConfig

MEDIA_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp", ".avi", ".m4v", ".mkv", ".mov", ".mp4"}


def _macos_system_alias_target(path: Path) -> Path | None:
    """Recognize only the root-owned macOS temporary-directory aliases."""
    if sys.platform != "darwin" or str(path) not in {"/var", "/tmp"}:
        return None
    target = Path("/private") / str(path).lstrip("/")
    parent = path.parent.lstat()
    if path.lstat().st_uid != 0 or parent.st_uid != 0 or parent.st_mode & 0o022:
        return None
    if path.readlink() not in {target, target.relative_to("/")}:
        return None
    return target


def _safe_path(path: Path) -> Path:
    path = path.absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            target = _macos_system_alias_target(part)
            if target is not None and part != path:
                # Recheck the entire physical path; links beneath /private are
                # not trusted. Never accept the system alias itself as a root.
                return _safe_path(target / path.relative_to(part))
            raise ValueError(f"Refusing linked data path: {part}")
        if part.exists():
            info = part.lstat()
            if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"Refusing reparse point: {part}")
    return path.resolve()


def _restrict_access(path: Path) -> None:
    path = _safe_path(path)
    if path.is_file() and path.stat().st_nlink != 1:
        raise ValueError(f"Refusing multiply-linked data file: {path}")
    if os.name != "nt":
        path.chmod(0o700 if path.is_dir() else 0o600)
        return
    # Build a fresh DACL: do not retain existing broad explicit grants.
    script = r"""
$ErrorActionPreference = 'Stop'
$target = Get-Item -LiteralPath $env:MONITOR_ACL_TARGET -Force
$owner = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$system = [System.Security.Principal.SecurityIdentifier]::new('S-1-5-18')
if ($target.PSIsContainer) {
    $acl = [System.Security.AccessControl.DirectorySecurity]::new()
    $inherit = [System.Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit'
} else {
    $acl = [System.Security.AccessControl.FileSecurity]::new()
    $inherit = [System.Security.AccessControl.InheritanceFlags]::None
}
$acl.SetOwner($owner)
$acl.SetAccessRuleProtection($true, $false)
foreach ($sid in @($owner, $system)) {
    $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $sid, 'FullControl', $inherit, 'None', 'Allow')
    $acl.AddAccessRule($rule)
}
$target.SetAccessControl($acl)
"""
    environment = os.environ.copy()
    environment["MONITOR_ACL_TARGET"] = str(path)
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def ensure_private_directory(path: Path) -> None:
    """Protect newly created directories before writing data into them.

    Existing directories require the explicit --secure-data migration command.
    """
    path = _safe_path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Not a data directory: {path}")
        return
    ensure_private_directory(path.parent)
    path.mkdir(mode=0o700)
    try:
        _restrict_access(path)
    except Exception:
        # Do not leave an unprotected directory looking successfully initialized.
        path.rmdir()
        raise


def safe_file_path(path: Path) -> Path:
    """Validate the original spelling before resolving links or opening a file."""
    path = _safe_path(path)
    if path.exists():
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Refusing non-regular or multiply-linked file: {path}")
    return path


def open_private_append(path: Path) -> TextIOWrapper:
    """Open a regular log/event file without truncation or following final links."""
    path = safe_file_path(path)
    ensure_private_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or not os.path.samestat(info, safe_file_path(path).stat())
        ):
            raise ValueError(f"Data file changed while opening: {path}")
        _restrict_access(path)
        return TextIOWrapper(FileIO(descriptor, mode="ab"), encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def private_temporary_file(directory: Path, suffix: str = "") -> Iterator[Path]:
    """Reserve an unpredictable private file; preserve suffix for media codecs."""
    ensure_private_directory(directory)
    directory = _safe_path(directory)
    descriptor, name = tempfile.mkstemp(prefix=".monitor-", suffix=suffix, dir=directory)
    os.close(descriptor)
    path = Path(name)
    try:
        _restrict_access(path)
        yield path
    finally:
        safe_file_path(path).unlink(missing_ok=True)


@contextmanager
def atomic_output(destination: Path) -> Iterator[Path]:
    """Publish only complete output after rechecking both paths and file types."""
    destination = safe_file_path(destination)
    with private_temporary_file(destination.parent, destination.suffix) as temporary:
        yield temporary
        safe_file_path(temporary)
        safe_file_path(destination)
        os.replace(temporary, destination)


@dataclass(frozen=True)
class RetentionRule:
    category: str
    root: Path
    days: int

    def __post_init__(self) -> None:
        if self.category not in {"recordings", "identities", "events", "logs"}:
            raise ValueError("Unknown retention category")
        if not isinstance(self.days, int) or self.days < 1:
            raise ValueError("Retention days must be a positive integer")

    def matches(self, path: Path) -> bool:
        if self.category == "recordings":
            return path.stem.endswith("_annotated") and path.suffix.lower() in MEDIA_SUFFIXES
        if self.category == "identities":
            return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if self.category == "events":
            return path.suffix.lower() == ".jsonl"
        return re.fullmatch(r"monitoring\.log(?:\.[0-9]+)?", path.name) is not None


@dataclass(frozen=True)
class ExpiredFile:
    path: Path
    root: Path
    category: str
    modified_ns: int
    size: int
    inode: int


def retention_rules(config: AppConfig) -> list[RetentionRule]:
    return [
        RetentionRule("recordings", config.output_dir, config.recording_retention_days),
        RetentionRule("identities", config.enrollments_dir, config.identity_retention_days),
        RetentionRule("events", config.event_dir, config.log_retention_days),
        RetentionRule("logs", config.log_dir, config.log_retention_days),
    ]


def _walk(root: Path):
    if not root.exists():
        return

    def fail(error: OSError) -> None:
        raise error

    for directory, directories, files in os.walk(root, followlinks=False, onerror=fail):
        for name in directories + files:
            path = _safe_path(Path(directory) / name)
            if not path.is_relative_to(root):
                raise ValueError("Data path escaped its managed root")
            yield path


def plan_retention(rules: list[RetentionRule], now: float | None = None) -> list[ExpiredFile]:
    now = time.time() if now is None else now
    result: dict[Path, ExpiredFile] = {}
    for rule in rules:
        root = _safe_path(rule.root)
        for path in _walk(root):
            if not path.is_file() or not rule.matches(path):
                continue
            info = path.stat()
            if info.st_nlink != 1:
                raise ValueError(f"Refusing multiply-linked data file: {path}")
            if info.st_mtime < now - rule.days * 86400:
                result[path] = ExpiredFile(
                    path, root, rule.category, info.st_mtime_ns, info.st_size, info.st_ino
                )
    return sorted(result.values(), key=lambda item: str(item.path))


def apply_retention(plan: list[ExpiredFile]) -> int:
    # Validate the complete preview before deleting anything.
    for item in plan:
        path = _safe_path(item.path)
        if not path.is_relative_to(_safe_path(item.root)) or path == item.root:
            raise ValueError("Retention path escaped its managed root")
        info = path.stat()
        if (info.st_mtime_ns, info.st_size, info.st_ino, info.st_nlink) != (
            item.modified_ns,
            item.size,
            item.inode,
            1,
        ):
            raise ValueError(f"File changed since retention preview: {path}")
    for item in plan:
        # No recursive delete; recheck parent links immediately before unlink.
        _safe_path(item.path).unlink()
    # Empty identity folders can also reveal names. Remove only empty parents.
    for item in plan:
        if item.category == "identities":
            parent = item.path.parent
            while parent != item.root and parent.is_relative_to(item.root):
                try:
                    _safe_path(parent).rmdir()
                except OSError:
                    break
                parent = parent.parent
    return len(plan)


def secure_data(config: AppConfig) -> None:
    roots = [config.output_dir, config.enrollments_dir]
    roots.extend([config.event_dir, config.log_dir])
    paths: set[Path] = set()
    for root in roots:
        root = _safe_path(root)
        if root in {Path(root.anchor), Path.home().resolve(), config.project_root.resolve()}:
            raise ValueError("Refusing to change permissions on a broad root directory")
        ensure_private_directory(root)
        paths.add(root)
        paths.update(_walk(root))
    for path in sorted(paths, key=lambda item: len(item.parts)):
        _restrict_access(path)
