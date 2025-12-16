"""CLI for lance-memtest."""

import sys
from memtest import get_library_path


def main():
    """Main CLI entry point."""
    args = sys.argv[1:]

    if not args or args[0] == "path":
        lib_path = get_library_path()
        if lib_path is None:
            print(
                "lance-memtest is not supported on this platform",
                file=sys.stderr,
            )
            return 1
        print(lib_path)
        return 0
    else:
        print(f"Unknown command: {args[0]}", file=sys.stderr)
        print("Usage: lance-memtest [path]", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
