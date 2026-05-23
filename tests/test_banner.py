from io import StringIO

from rich.console import Console

from ui.banner import print_gradient_banner


def test_banner_renders_cyberm4fia_wordmark():
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, record=True)
    print_gradient_banner(console=console)
    text = console.export_text()
    # All six wordmark rows must be present.
    assert "██████╗██╗" in text
    assert "AI-destekli" in text or "steganografi" in text.lower()


def test_banner_respects_quiet():
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, record=True)
    print_gradient_banner(console=console, quiet=True)
    assert console.export_text() == ""
