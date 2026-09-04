"""Compact product wordmark for interactive CLI output."""
from rich.console import Console

_BANNER = r"""
 ____ _____ _____ ____    _    _   _  ___   ____ ____      _    ____  _   ___   __
/ ___|_   _| ____/ ___|  / \  | \ | |/ _ \ / ___|  _ \    / \  |  _ \| | | \ \ / /
\___ \ | | |  _|| |  _  / _ \ |  \| | | | | |  _| |_) |  / _ \ | |_) | |_| |\ V /
 ___) || | | |__| |_| |/ ___ \| |\  | |_| | |_| |  _ <  / ___ \|  __/|  _  | | |
|____/ |_| |_____\____/_/   \_\_| \_|\___/ \____|_| \_\/_/   \_\_|   |_| |_| |_|
"""

_DESCRIPTION = "Local-first steganography & DFIR workbench"

_default_console = Console(record=True)


def print_gradient_banner(*, console: Console | None = None, quiet: bool = False) -> None:
    """Print the STEGANOGRAPHY wordmark and product description."""
    if quiet:
        return
    out = console or _default_console
    lines = _BANNER.strip("\n").split("\n")
    start, end = (230, 230, 230), (40, 40, 40)
    for i, line in enumerate(lines):
        ratio = i / max(len(lines) - 1, 1)
        r, g, b = (int(start[j] + (end[j] - start[j]) * ratio) for j in range(3))
        out.print(f"[#{r:02x}{g:02x}{b:02x}]{line}[/]")
    out.print(f"[dim white]{_DESCRIPTION}[/]")
    out.print(f"[#646464]{'─' * 80}[/]\n")
