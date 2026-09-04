from io import StringIO

from rich.console import Console

from ui.banner import print_gradient_banner


def test_banner_renders_product_wordmark():
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, record=True)
    print_gradient_banner(console=console)
    text = console.export_text()
    assert "STEGANOGRAPHY" in text.replace(" ", "") or "____" in text
    assert "Local-first steganography & DFIR workbench" in text


def test_banner_respects_quiet():
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, record=True)
    print_gradient_banner(console=console, quiet=True)
    assert console.export_text() == ""
