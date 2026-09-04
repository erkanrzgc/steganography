import asyncio
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
from textual.widgets import Checkbox, Input, Select, Static

from core.database import _SCHEMA_V1, Database
from core.pipeline import AnalysisPipeline, PipelineReport
from core.result import AnalysisResult, FileAnalysis, FileInfo, Signal
from core.workspace import create_workspace, default_state_dir
from ui.tui import (
    AdvancedScreen,
    CasesScreen,
    HideScreen,
    HomeScreen,
    QuickScanScreen,
    RecoverScreen,
    SteganographyTUI,
    _format_report,
    _method,
    _paths,
)


class InstantPipeline(AnalysisPipeline):
    def analyze(self, path: Path) -> PipelineReport:
        result = AnalysisResult(
            "test_analyzer",
            55,
            (Signal("test_signal", 55, "synthetic heuristic", category="test"),),
            "synthetic",
        )
        analysis = FileAnalysis(
            FileInfo(
                str(path),
                path.name,
                path.stat().st_size,
                "a" * 64,
                "text",
                "text/plain",
                path.suffix,
                False,
            ),
            55,
            "medium",
            "balanced",
            (result,),
        )
        return PipelineReport("report", "suspicious", 0.55, analysis, (), 1.0)


def _content(screen, selector: str) -> str:
    return str(screen.query_one(selector, Static).content)


def test_tui_quick_scan_navigation_preserve_and_help(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("ordinary local evidence\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "two.txt").write_text("two")
    assert _paths(f"{source};{nested}") == [source, nested / "two.txt"]

    async def scenario():
        app = SteganographyTUI(
            state_dir=tmp_path / "state",
            pipeline_factory=lambda _: InstantPipeline(),
        )
        async with app.run_test(size=(100, 32)) as pilot:
            assert isinstance(app.screen, HomeScreen)
            await pilot.press("?")
            await pilot.pause()
            assert app.screen.query_one("#help-close")
            await pilot.click("#help-close")
            await pilot.click("#task-scan")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, QuickScanScreen)
            screen.query_one("#scan-paths", Input).value = str(source)
            screen.start_scan()
            for _ in range(20):
                await pilot.pause(0.05)
                if screen.reports:
                    break
            assert screen.reports
            assert "temporary" in _content(screen, "#scan-status")
            assert app.workspace.cases.list_cases() == []

            screen.query_one("#preserve-case-name", Input).value = "TUI Case"
            screen.preserve()
            assert "required" in _content(screen, "#preserve-status")
            screen.query_one("#preserve-password", Input).value = "vault-password"
            screen.preserve()
            assert "Preserved 1" in _content(screen, "#preserve-status")
            assert app.workspace.cases.list_cases()[0]["evidence_count"] == 1
            assert not app.workspace.vault.unlocked

            size = type("Size", (), {"width": 70, "height": 20})()
            screen.on_resize(type("Resize", (), {"size": size})())
            assert screen.query_one("#terminal-warning", Static).display
            screen.cancel_scan()
            await pilot.press("escape")
            assert isinstance(app.screen, HomeScreen)

    asyncio.run(scenario())


def test_tui_guided_studio_roundtrip_wrong_password_and_no_overwrite(tmp_path: Path):
    cover = tmp_path / "cover.png"
    Image.fromarray(np.random.default_rng(44).integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(
        cover
    )
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"guided payload")
    stego = tmp_path / "stego.png"
    recovered = tmp_path / "recovered.bin"

    async def scenario():
        app = SteganographyTUI(state_dir=tmp_path / "state")
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#task-hide")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, HideScreen)
            screen.query_one("#hide-carrier", Input).value = str(cover)
            screen.query_one("#hide-payload", Input).value = str(payload)
            screen.query_one("#hide-output", Input).value = str(stego)
            screen.capacity()
            assert "bytes available" in _content(screen, "#capacity-result")
            screen.hide()
            assert "not confidential" in _content(screen, "#hide-status")
            screen.query_one("#unencrypted-confirm", Checkbox).value = True
            screen.query_one("#hide-password", Input).value = "payload-password"
            screen.hide()
            assert stego.is_file()
            assert "encrypted" in _content(screen, "#hide-status")
            screen.query_one("#hide-password", Input).value = "payload-password"
            screen.hide()
            assert "already exists" in _content(screen, "#hide-status")

            await pilot.press("escape")
            await pilot.click("#task-recover")
            await pilot.pause()
            recover_screen = app.screen
            assert isinstance(recover_screen, RecoverScreen)
            recover_screen.query_one("#recover-carrier", Input).value = str(stego)
            recover_screen.query_one("#recover-output", Input).value = str(recovered)
            recover_screen.preview()
            assert '"payload_version": 3' in _content(recover_screen, "#recover-status")
            recover_screen.query_one("#recover-password", Input).value = "wrong"
            recover_screen.recover()
            assert "authentication" in _content(recover_screen, "#recover-status")
            assert not recovered.exists()
            recover_screen.query_one("#recover-password", Input).value = "payload-password"
            recover_screen.recover()
            assert recovered.read_bytes() == payload.read_bytes()
            recover_screen.query_one("#recover-password", Input).value = "payload-password"
            recover_screen.recover()
            assert "already exists" in _content(recover_screen, "#recover-status")

    asyncio.run(scenario())


def test_tui_case_live_scan_export_advanced_and_helpers(tmp_path: Path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("case evidence\n")
    report_out = tmp_path / "report.json"

    async def scenario():
        workspace = create_workspace(tmp_path / "state")
        app = SteganographyTUI(workspace=workspace)
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#task-cases")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, CasesScreen)
            screen.query_one("#vault-password", Input).value = "vault-password"
            screen.unlock()
            assert workspace.vault.unlocked
            screen.query_one("#case-name", Input).value = "Live Case"
            screen.create_case()
            assert workspace.cases.list_cases()
            screen.query_one("#case-evidence", Input).value = str(evidence)
            screen.add_evidence()
            assert workspace.cases.list_cases()[0]["evidence_count"] == 1
            screen.scan()
            for _ in range(100):
                await pilot.pause(0.05)
                scans = workspace.scans.list_for_case(workspace.cases.list_cases()[0]["id"])
                if scans and scans[0]["status"] in {"completed", "failed", "cancelled"}:
                    break
            assert scans[0]["status"] == "completed"
            screen.query_one("#case-report-path", Input).value = str(report_out)
            screen.query_one("#case-report-format", Select).value = "json"
            screen.export()
            assert '"schema_version": "2.0"' in report_out.read_text()
            screen.export()
            assert "already exists" in _content(screen, "#case-status")
            screen.lock_vault()
            assert not workspace.vault.unlocked
            screen.query_one("#vault-password", Input).value = "wrong-password"
            screen.unlock()
            assert "failed" in _content(screen, "#case-status")
            screen.query_one("#vault-password", Input).value = "vault-password"
            screen.unlock()
            assert workspace.vault.unlocked
            screen.cancel()

            await pilot.press("escape")
            app.push_screen(AdvancedScreen())
            await pilot.pause()
            assert isinstance(app.screen, AdvancedScreen)
            assert len(app.screen.query("TabPane")) == 5

    asyncio.run(scenario())

    blank = Select([("Auto", "auto")], value="auto", allow_blank=False)
    assert _method(blank) is None
    report = InstantPipeline().analyze(evidence)
    text = _format_report(report)
    assert "SUSPICIOUS" in text and "False-positive" in text

    unavailable = AnalysisResult(
        "optional_tool", 0, (), None, status="unavailable", error="not installed"
    )
    unavailable_analysis = FileAnalysis(
        report.analysis.file,
        0,
        "low",
        "balanced",
        (unavailable,),
    )
    unavailable_report = PipelineReport(
        "unavailable",
        "inconclusive",
        0.0,
        unavailable_analysis,
        (),
        1.0,
    )
    assert "Unavailable tools" in _format_report(unavailable_report)

    cancel_workspace = create_workspace(tmp_path / "cancel-state")
    cancel_workspace.vault.initialize("cancel-password")
    cancel_case = cancel_workspace.cases.create_case("Cancel Case")
    cancel_workspace.cases.add_evidence(cancel_case["id"], evidence)
    queued = cancel_workspace.scans.create(cancel_case["id"], profile="balanced")
    cancelled = cancel_workspace.scans.run(queued["id"], should_cancel=lambda: True)
    assert cancelled["status"] == "cancelled"


def test_workspace_state_environment_and_cli_entry(tmp_path: Path, monkeypatch, capsys):
    from cli import main

    configured = tmp_path / "configured"
    monkeypatch.setenv("STEGANO_STATE_DIR", str(configured))
    assert default_state_dir() == configured
    monkeypatch.delenv("STEGANO_STATE_DIR")
    assert default_state_dir().name == "steganography"

    launched: list[Path | str | None] = []
    monkeypatch.setattr("ui.tui.run_tui", launched.append)
    assert main(["tui", "--state-dir", str(tmp_path / "tui-state")]) == 0
    assert launched == [str(tmp_path / "tui-state")]

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert main([]) == 0
    assert "usage: steganography" in capsys.readouterr().out


def test_database_migrates_scan_cancellation_status(tmp_path: Path):
    path = tmp_path / "v1.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(_SCHEMA_V1.replace(",'cancelled'", ""))
    connection.execute("PRAGMA user_version=1")
    connection.execute(
        "INSERT INTO cases(id,name,created_at,updated_at) VALUES('c','case','now','now')"
    )
    connection.execute(
        "INSERT INTO scans(id,case_id,status,profile,created_at,updated_at) "
        "VALUES('s','c','queued','balanced','now','now')"
    )
    connection.commit()
    connection.close()

    database = Database(path)
    with database.transaction() as migrated:
        migrated.execute("UPDATE scans SET status='cancelled' WHERE id='s'")
    with database.connect() as migrated:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            migrated.execute("SELECT status FROM scans WHERE id='s'").fetchone()[0] == "cancelled"
        )
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
