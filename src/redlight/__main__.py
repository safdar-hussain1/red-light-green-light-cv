"""Lets `python -m redlight` run the CLI, same as the installed `redlight` script."""

from redlight.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
