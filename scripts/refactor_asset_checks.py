"""Check optional Pi assets without deleting local models or importing tasks early."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = [
    ROOT / "src/assets/robots/pi" / name
    for name in ("pi.xml", "pi_12dof_release_v1.urdf")
]


def main():
    # Factory imports must never register tasks, even if assets are present.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import dme_thubo.env_cfg,sys; assert 'src.tasks' not in sys.modules",
        ],
        cwd=ROOT,
        check=True,
    )
    hidden = []
    try:
        for asset in ASSETS:
            assert asset.resolve().is_relative_to(ROOT.resolve())
            if asset.exists():
                backup = asset.with_name(asset.name + ".refactor-check")
                if backup.exists():
                    raise FileExistsError(backup)
                asset.rename(backup)
                hidden.append((asset, backup))
        code = """
import os
os.environ.update(HOST_NUM_ENVS='invalid', HOST_ACTION_MODE='invalid', HOST_POSES='invalid')
import src.tasks
from mjlab.tasks.registry import list_tasks
from dme_thubo.env_cfg import dme_thubo_env_cfg
assert 'THUBO-Flat-Stand' in list_tasks()
assert 'HoST-GetUp-THUBO' in list_tasks()
assert 'HoST-GetUp-PI' not in list_tasks()
assert dme_thubo_env_cfg(task='host_getup').scene.entities['robot']
try:
    dme_thubo_env_cfg(task='host_getup', robot='pi')
except FileNotFoundError as error:
    assert 'pi.xml' in str(error) and 'convert_pi.py' in str(error)
else:
    raise AssertionError('Missing Pi must fail explicitly')
"""
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    finally:
        for asset, backup in reversed(hidden):
            backup.rename(asset)
    for asset in ASSETS:
        path = asset.relative_to(ROOT).as_posix()
        subprocess.run(["git", "check-ignore", "--", path], cwd=ROOT, check=True)
        tracked = subprocess.check_output(["git", "ls-files", "--", path], cwd=ROOT)
        assert not tracked, path
    print(
        "PASS: pure factory import, missing-Pi registration/errors, local copies restored, Git ignore/untracked"
    )


if __name__ == "__main__":
    main()
