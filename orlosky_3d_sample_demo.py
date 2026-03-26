"""
Simple launcher for the Orlosky/JEOresearch EyeTracker demo.

Runs the detector on the repository's sample eye video:
  eyetracker/eye_test.mp4

This avoids the original script's hardcoded Windows path + file dialog flow.
"""

from pathlib import Path
import sys
import types


def _ensure_tkinter_stub():
    """
    OrloskyPupilDetector imports tkinter at module import time.
    Some Python builds (like current one) may not include _tkinter.
    For this demo wrapper we only call process_video(), so a lightweight
    stub is enough to satisfy imports.
    """
    try:
        import tkinter  # noqa: F401
        return
    except Exception:
        pass

    tk_stub = types.ModuleType("tkinter")
    ttk_stub = types.ModuleType("tkinter.ttk")
    filedialog_stub = types.ModuleType("tkinter.filedialog")

    class _DummyTk:
        def withdraw(self):
            return None

    def _dummy_askopenfilename(*_args, **_kwargs):
        return ""

    tk_stub.Tk = _DummyTk
    filedialog_stub.askopenfilename = _dummy_askopenfilename

    sys.modules["tkinter"] = tk_stub
    sys.modules["tkinter.ttk"] = ttk_stub
    sys.modules["tkinter.filedialog"] = filedialog_stub


def main():
    repo_dir = Path(__file__).resolve().parent / "eyetracker"
    if not repo_dir.exists():
        raise FileNotFoundError(f"Missing repo folder: {repo_dir}")

    sample_video = repo_dir / "eye_test.mp4"
    if not sample_video.exists():
        raise FileNotFoundError(f"Missing sample video: {sample_video}")

    # Import directly from cloned repo.
    sys.path.insert(0, str(repo_dir))
    _ensure_tkinter_stub()
    from OrloskyPupilDetector import process_video  # pylint: disable=import-error

    print(f"Running Orlosky sample demo on: {sample_video}")
    print("Controls: 'q' quit, SPACE pause/resume")
    process_video(str(sample_video), 1)


if __name__ == "__main__":
    main()
