"""Keyboard-first Textual workbench for guided and advanced workflows."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, cast

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual.worker import Worker, get_current_worker

from core.ctf import CTFLimits, CTFMode, CTFReport, CTFService
from core.pipeline import AnalysisPipeline, PipelineReport
from core.service import AnalysisService
from core.workspace import Workspace, create_workspace
from report.v2 import html_v2, json_v2, sarif_v2

SUPPORTED_FILES = "PNG, BMP, JPEG/TIFF, WAV PCM16, TXT/MD, GIF, PDF and ZIP"
VERDICT_HELP = {
    "confirmed": "A verified tool marker or successful extraction was found.",
    "likely": "Strong indicators were found; validate with extraction or another tool.",
    "suspicious": "Heuristics deserve review but may be a false positive.",
    "no_indicators": "No supported detector found an indicator; this is not proof of absence.",
    "inconclusive": "The available analyzers could not reach a reliable result.",
}


def _paths(value: str) -> list[Path]:
    """Parse semicolon-separated paths and expand directories one level recursively."""
    values: list[Path] = []
    for item in value.replace("\n", ";").split(";"):
        path = Path(item.strip()).expanduser()
        if not item.strip():
            continue
        if path.is_dir():
            values.extend(sorted(candidate for candidate in path.rglob("*") if candidate.is_file()))
        else:
            values.append(path)
    return values


def _method(select: Select[str]) -> str | None:
    value = select.value
    return None if value is Select.NULL or value == "auto" else str(value)


def _set_text(screen: Screen[Any], selector: str, value: str) -> None:
    screen.query_one(selector, Static).update(value)


class WorkbenchScreen(Screen[None]):
    """Common chrome and navigation bindings."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "back", "Back"),
        Binding("question_mark", "help", "Help"),
    ]

    def compose_chrome(self, *content: Widget) -> ComposeResult:
        yield Header()
        yield Static("", id="terminal-warning")
        yield from content
        yield Footer()

    def action_back(self) -> None:
        self.workbench.show_home()

    def action_help(self) -> None:
        self.app.push_screen(HelpScreen())

    @property
    def workbench(self) -> SteganographyTUI:
        return cast("SteganographyTUI", self.app)

    def on_resize(self, event: events.Resize) -> None:
        warning = self.query_one("#terminal-warning", Static)
        is_narrow = event.size.width < 80 or event.size.height < 24
        warning.update(
            "Terminal is narrow. Use at least 80×24 for the complete workspace."
            if is_narrow
            else ""
        )
        warning.display = is_narrow


class HomeScreen(WorkbenchScreen):
    BINDINGS = [Binding("question_mark", "help", "Help")]

    def compose(self) -> ComposeResult:
        yield from self.compose_chrome(
            ScrollableContainer(
                Static("STEGANOGRAPHY", classes="wordmark"),
                Static("Local-first steganography & DFIR workbench", classes="tagline"),
                Static("Choose a task. No operation uses the network by default."),
                Button("Scan suspicious files", id="task-scan", variant="primary"),
                Static("Analyze one or more files without copying them into evidence custody."),
                Button("Hide data", id="task-hide"),
                Static("Capacity-aware payload v3 embedding with optional encryption."),
                Button("Recover hidden data", id="task-recover"),
                Static("Auto-detect a supported carrier and write to a new output file."),
                Button("Manage evidence", id="task-cases"),
                Static("Encrypted vault, cases, persistent scans, and report export."),
                Button("Advanced workspace", id="task-advanced"),
                Button("Solve a CTF challenge", id="task-ctf"),
                Static("Run a bounded recovery playbook and inspect its artifact graph."),
                id="home-content",
            )
        )

    @on(Button.Pressed)
    def open_task(self, event: Button.Pressed) -> None:
        screens: dict[str, type[Screen[None]]] = {
            "task-scan": QuickScanScreen,
            "task-ctf": CTFScreen,
            "task-hide": HideScreen,
            "task-recover": RecoverScreen,
            "task-cases": CasesScreen,
            "task-advanced": AdvancedScreen,
        }
        screen = screens.get(event.button.id or "")
        if screen:
            self.app.push_screen(screen())


class CTFScreen(WorkbenchScreen):
    """Guided front end for the same bounded playbook used by CLI and API."""

    def __init__(self) -> None:
        super().__init__()
        self.ctf_worker: Worker[Any] | None = None

    def compose(self) -> ComposeResult:
        yield from self.compose_chrome(
            ScrollableContainer(
                Label("Solve a CTF challenge", classes="title"),
                Input(placeholder="Challenge file", id="ctf-input"),
                Input(placeholder="New output directory", id="ctf-output"),
                Select(
                    [("Quick", "quick"), ("Balanced", "balanced"), ("Deep", "deep")],
                    value="balanced",
                    allow_blank=False,
                    id="ctf-mode",
                ),
                Input(placeholder="Optional wordlist file", id="ctf-wordlist"),
                Input(placeholder="Optional extraction password", password=True, id="ctf-password"),
                Horizontal(
                    Button("Run playbook", id="ctf-run", variant="primary"),
                    Button("Cancel", id="ctf-cancel", disabled=True),
                ),
                ProgressBar(total=7, show_eta=False, id="ctf-progress"),
                Static(
                    "Ready. Defaults: depth 3, 256 artifacts, 1 GiB, 180 seconds.", id="ctf-status"
                ),
                Static("", id="ctf-result"),
                id="ctf-content",
            )
        )

    @on(Button.Pressed, "#ctf-run")
    def start(self) -> None:
        source = Path(self.query_one("#ctf-input", Input).value).expanduser()
        output = Path(self.query_one("#ctf-output", Input).value).expanduser()
        if not source.is_file() or output.exists():
            _set_text(self, "#ctf-status", "Choose an existing file and a new output directory.")
            return
        wordlist_value = self.query_one("#ctf-wordlist", Input).value.strip()
        wordlist = Path(wordlist_value).expanduser() if wordlist_value else None
        mode = str(self.query_one("#ctf-mode", Select).value)
        password = self.query_one("#ctf-password", Input).value or None
        self.query_one("#ctf-password", Input).value = ""
        self.query_one("#ctf-run", Button).disabled = True
        self.query_one("#ctf-cancel", Button).disabled = False
        self.query_one("#ctf-progress", ProgressBar).update(progress=0)
        self.ctf_worker = self.run_ctf(source, output, mode, wordlist, password)

    @work(thread=True, exclusive=True, group="ctf")
    def run_ctf(
        self,
        source: Path,
        output: Path,
        mode: str,
        wordlist: Path | None,
        password: str | None,
    ) -> CTFReport:
        worker = get_current_worker()

        def event(value: dict[str, Any]) -> None:
            if value.get("stage"):
                self.app.call_from_thread(self._ctf_stage, str(value["stage"]))

        report = CTFService().solve(
            source,
            output,
            mode=cast(CTFMode, mode),
            wordlist=wordlist,
            password=password,
            limits=CTFLimits(),
            should_cancel=lambda: worker.is_cancelled,
            on_event=event,
        )
        (output / "report.json").write_text(json_v2(report.to_dict()), encoding="utf-8")
        self.app.call_from_thread(self._ctf_finished, report)
        return report

    def _ctf_stage(self, stage: str) -> None:
        stages = {
            "identify": 1,
            "metadata_structure": 2,
            "native_detectors": 3,
            "external_tools": 4,
            "carving_decoding_extraction": 5,
            "recursive_analysis": 6,
        }
        self.query_one("#ctf-progress", ProgressBar).update(progress=stages.get(stage, 0))
        _set_text(self, "#ctf-status", f"Playbook stage: {stage}")

    def _ctf_finished(self, report: CTFReport) -> None:
        self.query_one("#ctf-run", Button).disabled = False
        self.query_one("#ctf-cancel", Button).disabled = True
        self.query_one("#ctf-progress", ProgressBar).update(progress=7)
        _set_text(self, "#ctf-status", f"CTF job {report.status}: {report.verdict}")
        tree = [
            f"{'  ' * item.depth}└─ {item.name} [{item.provenance}]" for item in report.artifacts
        ]
        tools = [f"{item.tool}: {item.status}" for item in report.tools]
        _set_text(self, "#ctf-result", "\n".join([*tree, *tools]))

    @on(Button.Pressed, "#ctf-cancel")
    def cancel(self) -> None:
        if self.ctf_worker:
            self.ctf_worker.cancel()


class QuickScanScreen(WorkbenchScreen):
    """Ephemeral analysis. Source files remain in place until explicitly preserved."""

    def __init__(self) -> None:
        super().__init__()
        self.source_paths: list[Path] = []
        self.reports: list[PipelineReport] = []
        self.scan_worker: Worker[Any] | None = None

    def compose(self) -> ComposeResult:
        profile = Select(
            [
                ("Balanced — practical defaults", "balanced"),
                ("Sensitive — finds more, with more false positives", "sensitive"),
                ("Strict — reports only stronger signals", "strict"),
            ],
            value="balanced",
            allow_blank=False,
            id="scan-profile",
        )
        yield from self.compose_chrome(
            ScrollableContainer(
                Label("Scan suspicious files", classes="title"),
                Static(f"Supported: {SUPPORTED_FILES}"),
                Input(
                    placeholder="/path/file.png; /path/another.wav; /path/folder",
                    id="scan-paths",
                ),
                profile,
                Horizontal(
                    Button("Start quick scan", id="start-scan", variant="primary"),
                    Button("Cancel", id="cancel-scan", disabled=True),
                ),
                ProgressBar(total=100, show_eta=False, id="scan-progress"),
                Static(
                    "Quick Scan is temporary; source files are not copied to the vault.",
                    id="scan-status",
                ),
                Static("", id="scan-result"),
                Container(
                    Label("Preserve in a case"),
                    Input(placeholder="Case name", id="preserve-case-name"),
                    Input(placeholder="Vault password", password=True, id="preserve-password"),
                    Button("Preserve selected files", id="preserve-scan", disabled=True),
                    Static("", id="preserve-status"),
                    id="preserve-panel",
                ),
                id="quick-scan-content",
            )
        )

    @on(Button.Pressed, "#start-scan")
    def start_scan(self) -> None:
        paths = _paths(self.query_one("#scan-paths", Input).value)
        missing = [str(path) for path in paths if not path.is_file()]
        if not paths or missing:
            _set_text(self, "#scan-status", "Choose one or more existing files.")
            return
        profile = str(self.query_one("#scan-profile", Select).value)
        self.source_paths = paths
        self.reports = []
        self.query_one("#start-scan", Button).disabled = True
        self.query_one("#cancel-scan", Button).disabled = False
        self.query_one("#preserve-scan", Button).disabled = True
        self.query_one("#scan-progress", ProgressBar).update(progress=0)
        _set_text(self, "#scan-status", f"Analyzing 0/{len(paths)} files…")
        self.scan_worker = self.run_quick_scan(paths, profile)

    @work(thread=True, exclusive=True, group="quick-scan")
    def run_quick_scan(self, paths: list[Path], profile: str) -> list[PipelineReport]:
        worker = get_current_worker()
        reports: list[PipelineReport] = []
        pipeline = self.workbench.analysis_pipeline(profile)
        for index, path in enumerate(paths, start=1):
            if worker.is_cancelled:
                break
            reports.append(pipeline.analyze(path))
            self.app.call_from_thread(self._scan_progress, index, len(paths), path.name)
        self.app.call_from_thread(self._scan_finished, reports, worker.is_cancelled)
        return reports

    def _scan_progress(self, completed: int, total: int, name: str) -> None:
        self.query_one("#scan-progress", ProgressBar).update(
            progress=completed * 100 / max(total, 1)
        )
        _set_text(self, "#scan-status", f"Analyzing {completed}/{total}: {name}")

    def _scan_finished(self, reports: list[PipelineReport], cancelled: bool) -> None:
        self.reports = reports
        self.query_one("#start-scan", Button).disabled = False
        self.query_one("#cancel-scan", Button).disabled = True
        self.query_one("#preserve-scan", Button).disabled = not reports
        if cancelled:
            _set_text(self, "#scan-status", f"Cancelled after {len(reports)} file(s).")
        else:
            _set_text(
                self,
                "#scan-status",
                f"Completed {len(reports)} temporary analysis result(s).",
            )
        _set_text(self, "#scan-result", "\n\n".join(_format_report(item) for item in reports))

    @on(Button.Pressed, "#cancel-scan")
    def cancel_scan(self) -> None:
        if self.scan_worker:
            self.scan_worker.cancel()

    @on(Button.Pressed, "#preserve-scan")
    def preserve(self) -> None:
        password = self.query_one("#preserve-password", Input).value
        case_name = self.query_one("#preserve-case-name", Input).value.strip()
        if not password or not case_name:
            _set_text(self, "#preserve-status", "Case name and vault password are required.")
            return
        workspace = self.workbench.workspace
        try:
            if workspace.vault.initialized:
                workspace.vault.unlock(password)
            else:
                workspace.vault.initialize(password)
            case = workspace.cases.create_case(case_name)
            for path in self.source_paths:
                workspace.cases.add_evidence(case["id"], path)
            _set_text(
                self,
                "#preserve-status",
                f"Preserved {len(self.source_paths)} file(s) in case {case_name}.",
            )
        except Exception as exc:
            _set_text(self, "#preserve-status", f"Could not preserve evidence: {exc}")
        finally:
            workspace.lock()
            self.query_one("#preserve-password", Input).value = ""


class HideScreen(WorkbenchScreen):
    def compose(self) -> ComposeResult:
        yield from self.compose_chrome(
            ScrollableContainer(
                Label("Hide data", classes="title"),
                Static(
                    "Carrier formats: PNG/BMP, WAV, TXT/MD, JPEG/TIFF, GIF/PDF. "
                    "Recompression or format conversion can destroy hidden payloads."
                ),
                Input(placeholder="Carrier file", id="hide-carrier"),
                Input(placeholder="Payload file", id="hide-payload"),
                Input(placeholder="New output file (will not overwrite)", id="hide-output"),
                Select(_carrier_options(), value="auto", allow_blank=False, id="hide-method"),
                Input(placeholder="Optional placement key", password=True, id="hide-steg-key"),
                Input(
                    placeholder="Optional encryption password",
                    password=True,
                    id="hide-password",
                ),
                Checkbox("Compress payload", value=True, id="hide-compress"),
                Input(value="0", type="integer", id="hide-ecc"),
                Button("Check capacity", id="check-capacity"),
                Static("", id="capacity-result"),
                Checkbox(
                    "I understand that a passwordless payload is hidden but not confidential.",
                    id="unencrypted-confirm",
                ),
                Button("Hide payload", id="hide-run", variant="primary"),
                Static("Payload v3 is always used in Guided mode.", id="hide-status"),
                id="hide-content",
            )
        )

    @on(Button.Pressed, "#check-capacity")
    def capacity(self) -> None:
        try:
            carrier = Path(self.query_one("#hide-carrier", Input).value).expanduser()
            payload = Path(self.query_one("#hide-payload", Input).value).expanduser()
            result = self.workbench.workspace.studio.capacity(
                carrier, method=_method(self.query_one("#hide-method", Select))
            )
            methods = result["methods"]
            needed = payload.stat().st_size
            lines = [
                f"{item['method']}: {item['payload_v3_capacity_bytes']} bytes available; "
                f"{needed} bytes requested ({item['status']})."
                for item in methods
            ]
            _set_text(
                self,
                "#capacity-result",
                "\n".join(lines) or "No embedding method available.",
            )
        except Exception as exc:
            _set_text(self, "#capacity-result", f"Capacity check failed: {exc}")

    @on(Button.Pressed, "#hide-run")
    def hide(self) -> None:
        password = self.query_one("#hide-password", Input).value or None
        confirmed = self.query_one("#unencrypted-confirm", Checkbox).value
        if not password and not confirmed:
            _set_text(
                self,
                "#hide-status",
                "Warning: this would be hidden but not confidential. Check the confirmation first.",
            )
            return
        try:
            payload = Path(self.query_one("#hide-payload", Input).value).expanduser()
            carrier = Path(self.query_one("#hide-carrier", Input).value).expanduser()
            out = Path(self.query_one("#hide-output", Input).value).expanduser()
            capacity = self.workbench.workspace.studio.capacity(
                carrier, method=_method(self.query_one("#hide-method", Select))
            )
            available = max(
                (item["payload_v3_capacity_bytes"] for item in capacity["methods"]),
                default=0,
            )
            if payload.stat().st_size > available:
                raise ValueError(
                    f"payload needs {payload.stat().st_size} bytes; capacity is {available} bytes"
                )
            result = self.workbench.workspace.studio.embed(
                payload,
                carrier,
                out,
                password=password,
                method=_method(self.query_one("#hide-method", Select)),
                steg_key=self.query_one("#hide-steg-key", Input).value or None,
                no_clobber=True,
                compress=self.query_one("#hide-compress", Checkbox).value,
                ecc_symbols=int(self.query_one("#hide-ecc", Input).value or 0),
            )
            _set_text(
                self,
                "#hide-status",
                f"Wrote {result.out_path} via {result.carrier}; "
                + ("encrypted." if result.encrypted else "not encrypted."),
            )
        except Exception as exc:
            _set_text(self, "#hide-status", f"Hide failed: {exc}")
        finally:
            self.query_one("#hide-password", Input).value = ""
            self.query_one("#hide-steg-key", Input).value = ""


class RecoverScreen(WorkbenchScreen):
    def compose(self) -> ComposeResult:
        yield from self.compose_chrome(
            ScrollableContainer(
                Label("Recover hidden data", classes="title"),
                Input(placeholder="Carrier file", id="recover-carrier"),
                Input(placeholder="New output file (will not overwrite)", id="recover-output"),
                Select(_carrier_options(), value="auto", allow_blank=False, id="recover-method"),
                Input(placeholder="Optional password", password=True, id="recover-password"),
                Input(placeholder="Optional placement key", password=True, id="recover-steg-key"),
                Horizontal(
                    Button("Preview metadata", id="recover-preview"),
                    Button("Recover", id="recover-run", variant="primary"),
                ),
                Static("Automatic method detection is enabled.", id="recover-status"),
                id="recover-content",
            )
        )

    @on(Button.Pressed, "#recover-preview")
    def preview(self) -> None:
        try:
            value = self.workbench.workspace.studio.preview(
                Path(self.query_one("#recover-carrier", Input).value).expanduser(),
                method=_method(self.query_one("#recover-method", Select)),
                steg_key=self.query_one("#recover-steg-key", Input).value or None,
            )
            _set_text(self, "#recover-status", json.dumps(value, indent=2))
        except Exception as exc:
            _set_text(self, "#recover-status", f"Preview failed: {exc}")

    @on(Button.Pressed, "#recover-run")
    def recover(self) -> None:
        try:
            out = Path(self.query_one("#recover-output", Input).value).expanduser()
            size, carrier, version = self.workbench.workspace.studio.stego.extract_to(
                Path(self.query_one("#recover-carrier", Input).value).expanduser(),
                out,
                password=self.query_one("#recover-password", Input).value or None,
                method=_method(self.query_one("#recover-method", Select)),
                steg_key=self.query_one("#recover-steg-key", Input).value or None,
                no_clobber=True,
            )
            _set_text(
                self,
                "#recover-status",
                f"Recovered {size} bytes from payload v{version} via {carrier} to {out}.",
            )
        except Exception as exc:
            _set_text(self, "#recover-status", f"Recovery failed: {exc}")
        finally:
            self.query_one("#recover-password", Input).value = ""
            self.query_one("#recover-steg-key", Input).value = ""


class CasesScreen(WorkbenchScreen):
    def __init__(self) -> None:
        super().__init__()
        self.case_scan_worker: Worker[Any] | None = None
        self.active_scan_id: str | None = None

    def compose(self) -> ComposeResult:
        yield from self.compose_chrome(
            ScrollableContainer(
                Label("Manage cases", classes="title"),
                Input(placeholder="Vault password", password=True, id="vault-password"),
                Horizontal(
                    Button("Initialize / unlock vault", id="vault-unlock", variant="primary"),
                    Button("Lock", id="vault-lock"),
                ),
                Input(placeholder="New case name", id="case-name"),
                Button("Create case", id="case-create"),
                Select([], prompt="Select a case", id="case-select"),
                Input(placeholder="Evidence file to preserve", id="case-evidence"),
                Button("Preserve evidence", id="case-add"),
                Select(
                    [("Balanced", "balanced"), ("Sensitive", "sensitive"), ("Strict", "strict")],
                    value="balanced",
                    allow_blank=False,
                    id="case-profile",
                ),
                Horizontal(
                    Button("Run case scan", id="case-scan"),
                    Button("Cancel scan", id="case-cancel", disabled=True),
                ),
                ProgressBar(total=100, show_eta=False, id="case-progress"),
                Input(placeholder="Report output path", id="case-report-path"),
                Select(
                    [("HTML", "html"), ("JSON", "json"), ("SARIF", "sarif")],
                    value="html",
                    allow_blank=False,
                    id="case-report-format",
                ),
                Button("Export latest report", id="case-export"),
                Static("", id="case-status"),
                Static("", id="case-list"),
                id="cases-content",
            )
        )

    def on_mount(self) -> None:
        self.refresh_cases()

    def refresh_cases(self, selected: str | None = None) -> None:
        cases = self.workbench.workspace.cases.list_cases()
        select = self.query_one("#case-select", Select)
        select.set_options([(item["name"], item["id"]) for item in cases])
        if selected:
            select.value = selected
        _set_text(
            self,
            "#case-list",
            "\n".join(
                f"{item['name']}: {item['evidence_count']} evidence, {item['scan_count']} scans"
                for item in cases
            )
            or "No cases yet.",
        )

    def _case_id(self) -> str:
        value = self.query_one("#case-select", Select).value
        if value is Select.NULL:
            raise ValueError("select a case first")
        return str(value)

    @on(Button.Pressed, "#vault-unlock")
    def unlock(self) -> None:
        password = self.query_one("#vault-password", Input).value
        try:
            if self.workbench.workspace.vault.initialized:
                self.workbench.workspace.vault.unlock(password)
            else:
                self.workbench.workspace.vault.initialize(password)
            _set_text(self, "#case-status", "Vault unlocked for this session.")
        except Exception as exc:
            _set_text(self, "#case-status", f"Vault unlock failed: {exc}")
        finally:
            self.query_one("#vault-password", Input).value = ""

    @on(Button.Pressed, "#vault-lock")
    def lock_vault(self) -> None:
        self.workbench.workspace.lock()
        _set_text(self, "#case-status", "Vault locked.")

    @on(Button.Pressed, "#case-create")
    def create_case(self) -> None:
        try:
            case = self.workbench.workspace.cases.create_case(
                self.query_one("#case-name", Input).value
            )
            self.refresh_cases(case["id"])
            _set_text(self, "#case-status", f"Created case {case['name']}.")
        except Exception as exc:
            _set_text(self, "#case-status", f"Case creation failed: {exc}")

    @on(Button.Pressed, "#case-add")
    def add_evidence(self) -> None:
        try:
            value = self.workbench.workspace.cases.add_evidence(
                self._case_id(),
                Path(self.query_one("#case-evidence", Input).value).expanduser(),
            )
            self.refresh_cases(self._case_id())
            _set_text(self, "#case-status", f"Preserved {value['original_name']} in the vault.")
        except Exception as exc:
            _set_text(self, "#case-status", f"Evidence preservation failed: {exc}")

    @on(Button.Pressed, "#case-scan")
    def scan(self) -> None:
        try:
            case_id = self._case_id()
            profile = str(self.query_one("#case-profile", Select).value)
            scan = self.workbench.workspace.scans.create(case_id, profile=profile)
            self.active_scan_id = scan["id"]
            self.query_one("#case-scan", Button).disabled = True
            self.query_one("#case-cancel", Button).disabled = False
            self.case_scan_worker = self.run_case_scan(scan["id"])
        except Exception as exc:
            _set_text(self, "#case-status", f"Scan could not start: {exc}")

    @work(thread=True, exclusive=True, group="case-scan")
    def run_case_scan(self, scan_id: str) -> dict[str, Any]:
        worker = get_current_worker()

        def progress(done: int, total: int) -> None:
            self.app.call_from_thread(self._case_progress, done, total)

        result = self.workbench.workspace.scans.run(
            scan_id,
            progress_callback=progress,
            should_cancel=lambda: worker.is_cancelled,
        )
        self.app.call_from_thread(self._case_finished, result)
        return result

    def _case_progress(self, done: int, total: int) -> None:
        self.query_one("#case-progress", ProgressBar).update(progress=done * 100 / max(1, total))
        _set_text(self, "#case-status", f"Case scan: {done}/{total} files.")

    def _case_finished(self, result: dict[str, Any]) -> None:
        self.query_one("#case-scan", Button).disabled = False
        self.query_one("#case-cancel", Button).disabled = True
        _set_text(self, "#case-status", f"Case scan {result['status']}.")
        self.refresh_cases(result["case_id"])

    @on(Button.Pressed, "#case-cancel")
    def cancel(self) -> None:
        if self.case_scan_worker:
            self.case_scan_worker.cancel()

    @on(Button.Pressed, "#case-export")
    def export(self) -> None:
        try:
            scans = self.workbench.workspace.scans.list_for_case(self._case_id())
            completed = next((item for item in scans if item["result"]), None)
            if not completed:
                raise ValueError("no completed report is available")
            fmt = str(self.query_one("#case-report-format", Select).value)
            report = completed["result"]
            if fmt == "html":
                content = html_v2(report)
            elif fmt == "sarif":
                content = json.dumps(sarif_v2(report), indent=2)
            else:
                content = json_v2(report)
            out = Path(self.query_one("#case-report-path", Input).value).expanduser()
            if out.exists():
                raise FileExistsError(f"output already exists: {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")
            _set_text(self, "#case-status", f"Exported {fmt.upper()} report to {out}.")
        except Exception as exc:
            _set_text(self, "#case-status", f"Report export failed: {exc}")


class AdvancedScreen(WorkbenchScreen):
    def compose(self) -> ComposeResult:
        models = self.workbench.workspace.models.list()
        runtime = self.workbench.workspace.models.runtime_status()
        audit_valid, audit_events, audit_error = self.workbench.workspace.database.verify_audit()
        yield Header()
        yield Static("", id="terminal-warning")
        with TabbedContent(id="advanced-tabs"):
            with TabPane("Cases / Evidence / Scans", id="advanced-cases"):
                yield Static(
                    "Use Manage cases for custody operations. Persistent records and raw "
                    "findings remain available in the local SQLite workspace.\n\n"
                    "CTF playbook stages: identify → metadata/structure → native detectors → "
                    "external tools → carving/decoder graph → extraction → recursive analysis. "
                    "The guided CTF screen exposes live stages, tool status, artifact tree, "
                    "cancellation, and clean-directory reruns."
                )
            with TabPane("Studio", id="advanced-studio"):
                yield Static(
                    "Advanced controls map directly to method, channel, compression, ECC, "
                    "payload-version and placement-key core options."
                )
            with TabPane("Models", id="advanced-models"):
                yield Static(json.dumps({"installed": models, "runtime": runtime}, indent=2))
            with TabPane("Research", id="advanced-research"):
                yield Static(
                    "Dataset import and held-out benchmark are available. Training and ONNX "
                    "export require an explicit experiment configuration and are not run here."
                )
            with TabPane("Doctor / Settings", id="advanced-doctor"):
                yield Static(
                    json.dumps(
                        {
                            "state_dir": str(self.workbench.workspace.state_dir),
                            "vault_initialized": self.workbench.workspace.vault.initialized,
                            "audit_valid": audit_valid,
                            "audit_events": audit_events,
                            "audit_error": audit_error,
                            "network_default": "disabled",
                        },
                        indent=2,
                    )
                )
        yield Footer()


class HelpScreen(Screen[None]):
    BINDINGS = [Binding("escape", "dismiss", "Close"), Binding("question_mark", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Keyboard help", classes="title"),
            Static(
                "Tab / Shift+Tab: move focus\nEnter / Space: activate\nEsc: back\n?: help\n"
                "Ctrl+C: stop the application"
            ),
            Button("Close", id="help-close", variant="primary"),
            id="help-dialog",
        )

    @on(Button.Pressed, "#help-close")
    def close(self) -> None:
        self.dismiss()


def _carrier_options() -> list[tuple[str, str]]:
    return [
        ("Automatic (recommended)", "auto"),
        ("Image LSB", "image_lsb"),
        ("Scattered image LSB", "image_lsb_scatter"),
        ("WAV sample LSB", "audio_wav"),
        ("Text whitespace", "text_whitespace"),
        ("Text zero-width", "text_zerowidth"),
        ("JPEG/TIFF EXIF", "filestruct_exif"),
        ("JPEG marker", "image_jpeg"),
        ("GIF/PDF trailer", "filestruct_trailer"),
        ("Experimental JPEG DCT", "image_jpeg_dct"),
    ]


def _format_report(report: PipelineReport) -> str:
    rows = [
        f"{report.analysis.file.name}: {report.verdict.upper()} "
        f"({report.confidence:.0%} confidence)",
        VERDICT_HELP[report.verdict],
    ]
    if report.findings:
        for finding in report.findings:
            location = finding.region or (
                f"offset {finding.offset_start}–{finding.offset_end}"
                if finding.offset_start is not None
                else "location not reported"
            )
            rows.append(f"• {finding.analyzer}: {finding.detail} [{finding.status}; {location}]")
    else:
        rows.append("No supported analyzer reported a finding.")
    unavailable = [
        item for item in report.analysis.results if item.status in {"unavailable", "unsupported"}
    ]
    if unavailable:
        rows.append(
            "Unavailable tools: "
            + ", ".join(f"{item.analyzer} ({item.error or item.status})" for item in unavailable)
        )
    rows.append(
        "False-positive note: heuristic findings need independent validation. "
        "Next: inspect the named region, attempt recovery, or preserve the file in a case."
    )
    return "\n".join(rows)


class SteganographyTUI(App[None]):
    """Local-first guided terminal application."""

    TITLE = "Steganography"
    SUB_TITLE = "Local-first steganography & DFIR workbench"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { background: #0b0f14; color: #e7edf3; }
    Header { background: #111923; color: #d9ff65; }
    Footer { background: #111923; }
    ScrollableContainer { padding: 1 2; }
    .wordmark { color: #d9ff65; text-style: bold; text-align: center; padding: 1; }
    .tagline { text-align: center; color: #aab6c4; margin-bottom: 1; }
    .title { text-style: bold; color: #d9ff65; margin-bottom: 1; }
    Button { margin: 1 1 0 0; min-width: 24; }
    Button:focus, Input:focus, Select:focus, Checkbox:focus { border: tall #d9ff65; }
    Input, Select { margin: 1 0 0 0; }
    Horizontal { height: auto; }
    ProgressBar { margin: 1 0; }
    #terminal-warning { display: none; background: #6b4200; color: white; text-align: center; }
    #scan-result, #case-list, #capacity-result { margin-top: 1; padding: 1; background: #111923; }
    #preserve-panel { border: round #52606d; padding: 1; margin-top: 1; height: auto; }
    #help-dialog {
        width: 64; height: auto; padding: 2; border: round #d9ff65;
        background: #111923; align: center middle;
    }
    HelpScreen { align: center middle; background: rgba(0, 0, 0, 0.75); }
    TabPane { padding: 1 2; }
    """

    def __init__(
        self,
        *,
        state_dir: Path | str | None = None,
        workspace: Workspace | None = None,
        pipeline_factory: Callable[[str], AnalysisPipeline] | None = None,
    ) -> None:
        super().__init__()
        self.workspace = workspace or create_workspace(state_dir)
        self._pipeline_factory = pipeline_factory or (
            lambda profile: AnalysisPipeline(AnalysisService(profile=profile, ai_provider=None))
        )

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())

    def analysis_pipeline(self, profile: str) -> AnalysisPipeline:
        return self._pipeline_factory(profile)

    def show_home(self) -> None:
        while len(self.screen_stack) > 1:
            self.pop_screen()
        if not isinstance(self.screen, HomeScreen):
            self.push_screen(HomeScreen())

    def on_unmount(self) -> None:
        self.workspace.lock()


def run_tui(state_dir: Path | str | None = None) -> None:
    """Launch the terminal UI without starting an HTTP server."""
    SteganographyTUI(state_dir=state_dir).run()


__all__ = ["SteganographyTUI", "run_tui"]
