"""
Scikit-learn environment tracking: pip freeze, conda export patterns
"""

import subprocess
import sys
import os


def main():
    print("=" * 60)
    print("Environment Tracking: pip freeze, conda export")
    print("=" * 60)

    print("\n[1] Python version:")
    print(f"    {sys.version}")

    print("\n[2] pip freeze (current environment):")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [l for l in result.stdout.strip().split("\n") if "scikit" in l.lower() or "numpy" in l.lower()]
        for line in lines[:5]:
            print(f"    {line}")
        if len(lines) > 5:
            print(f"    ... ({len(lines)} total relevant packages)")
    except Exception as e:
        print(f"    Error: {e}")

    print("\n[3] Save requirements.txt pattern:")
    print("    pip freeze > requirements.txt")
    print("    pip install -r requirements.txt")

    print("\n[4] conda export pattern:")
    print("    conda env export > environment.yml")
    print("    conda env create -f environment.yml")

    print("\n[5] Minimal requirements for sklearn model:")
    deps = ["scikit-learn", "numpy", "joblib"]
    print(f"    Core: {', '.join(deps)}")
    print("    Add scipy if using sparse matrices")

    print("\n[6] Version pinning example:")
    print("    scikit-learn==1.3.2")
    print("    numpy>=1.24,<2.0")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
