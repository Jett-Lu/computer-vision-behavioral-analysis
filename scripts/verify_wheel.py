"""Install a built wheel outside the checkout and exercise its launcher/assets.

Reuse the caller's installed third-party dependencies; this checks distribution
contents and installation, not fresh dependency resolution or network access.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import venv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args()
    wheels = list(args.dist_dir.resolve().glob("integrated_monitoring_poc-*.whl"))
    if len(wheels) != 1:
        parser.error("dist_dir must contain exactly one application wheel")
    with tempfile.TemporaryDirectory(prefix="monitoring-wheel-") as directory:
        root = Path(directory).resolve()
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        site = Path(
            subprocess.check_output(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                text=True,
            ).strip()
        )
        # Also works when the caller itself uses a virtual environment.
        (site / "verification_dependencies.pth").write_text(
            sysconfig.get_path("purelib") + "\n", encoding="utf-8"
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--ignore-installed",
                str(wheels[0]),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        env.pop("MONITOR_HOME", None)
        subprocess.run(
            [
                str(python),
                "-c",
                """
from pathlib import Path
from config import AppConfig
assert AppConfig.from_env().project_root == Path.home() / '.integrated-monitoring-poc'
""",
            ],
            cwd=root,
            env=env,
            check=True,
        )
        env["MONITOR_HOME"] = str(root / "state")
        launcher = scripts / (
            "integrated-monitoring.exe" if os.name == "nt" else "integrated-monitoring"
        )
        subprocess.run(
            [str(launcher), "--help"], cwd=root, env=env, check=True, stdout=subprocess.DEVNULL
        )
        subprocess.run(
            [
                str(python),
                "-c",
                """
import os
import sys
from pathlib import Path
import config
import monitoring_assets
assert Path(config.__file__).is_relative_to(sys.prefix)
assert Path(monitoring_assets.__file__).is_relative_to(sys.prefix)
settings = config.AppConfig.from_env()
assert settings.project_root == Path(os.environ['MONITOR_HOME'])
assert settings.emotion_face_model_path.is_file()
assert settings.emotion_face_model_path.stat().st_size > 100_000
import detection
import emotion
import enrollment
import media_export
assert emotion.TMP_DIR.is_relative_to(settings.project_root)
assert enrollment.ENROLLMENTS_DIR == settings.enrollments_dir
assert media_export.DEFAULT_OUTPUT_DIR == settings.output_dir
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
options = vision.FaceDetectorOptions(base_options=python.BaseOptions(
    model_asset_path=str(settings.emotion_face_model_path)))
with vision.FaceDetector.create_from_options(options):
    pass
print('PASS: installed wheel launcher, bundled detector and writable runtime paths')
""",
            ],
            cwd=root,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
