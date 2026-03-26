import subprocess
import sys
from pathlib import Path


PORT = "COM8"
LOCAL_MAIN = Path(__file__).with_name("main.py")


def main():
    if not LOCAL_MAIN.exists():
        print(f"Could not find {LOCAL_MAIN}", file=sys.stderr)
        sys.exit(1)

    command = [
        "py",
        "-m",
        "mpremote",
        "connect",
        PORT,
        "fs",
        "cp",
        str(LOCAL_MAIN),
        ":main.py",
        "+",
        "reset",
    ]

    print(f"Copying {LOCAL_MAIN.name} to the Pico on {PORT}")
    print("If this fails, close any VS Code, MicroPico, Thonny, or serial monitor first.")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Deployment failed with exit code {exc.returncode}.", file=sys.stderr)
        sys.exit(exc.returncode)

    print("Deployment complete. The Pico was reset after copying main.py.")
    print("Next step: py test_pico_serial.py")


if __name__ == "__main__":
    main()
