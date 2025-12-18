from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, ProgressBar, RichLog
from textual.timer import Timer

from tui.views.base_view import BaseView
from tui.models import ProjectConfig

import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.status_manager import StatusManager
from src.file_paths import STATUS_DIR


class PipelineStep(Static):
    """Widget representing a single pipeline step with status icon."""
    
    STEP_ICONS = {
        "pending": "[ ]",
        "running": "[~]",
        "done": "[✓]",
        "error": "[✗]"
    }
    
    def __init__(self, step_name: str, step_index: int, **kwargs):
        super().__init__(**kwargs)
        self.step_name = step_name
        self.step_index = step_index
        self._status = "pending"
    
    def compose(self) -> ComposeResult:
        yield Static(f"{self.STEP_ICONS['pending']} {self.step_index + 1}. {self.step_name}", id=f"step-{self.step_index}")
    
    def set_status(self, status: str):
        """Set the step status: pending, running, done, error"""
        self._status = status
        icon = self.STEP_ICONS.get(status, "[ ]")
        
        # Update the label
        label = self.query_one(f"#step-{self.step_index}", Static)
        label.update(f"{icon} {self.step_index + 1}. {self.step_name}")
        
        # Update CSS classes for styling
        self.remove_class("step-pending", "step-running", "step-done", "step-error")
        self.add_class(f"step-{status}")


class PipelineView(BaseView):
    """PIPELINE STATUS view showing pipeline execution status."""
    
    POLL_INTERVAL = 2.0  # seconds
    HEARTBEAT_TIMEOUT = 60.0  # seconds - longer timeout for Google Drive sync delays
    
    def __init__(self, config: ProjectConfig, filepath: str | None = None):
        super().__init__(config, filepath)
        self._current_status_file: str | None = None
        self._last_run_id: str | None = None
        self._poll_timer: Timer | None = None
        self._is_pipeline_alive: bool = False
        self._displayed_log_count: int = 0  # Track how many logs we've displayed
    
    def compose(self) -> ComposeResult:
        # Determine config name for display
        config_name = os.path.basename(self.filepath) if self.filepath else "No config loaded"
        
        with Vertical(id="pipeline-container", classes="pipeline-container"):
            # Header showing current config
            with Horizontal(id="pipeline-header", classes="pipeline-header"):
                indicator = Static("●", id="alive-indicator", classes="alive-indicator alive-dead")
                indicator.tooltip = "● Green: Running (live)\n● Orange: Running (stale)\n● Red: Stopped/Error\n● Blue: Completed"
                yield indicator
                yield Static("Pipeline Status", classes="pipeline-title")
                yield Static(f"Config: {config_name}", id="config-label", classes="config-label")
            
            # Main content area
            with Horizontal(id="pipeline-main", classes="pipeline-main"):
                # Left panel - Steps
                with Vertical(id="pipeline-steps-panel", classes="pipeline-steps-panel"):
                    yield Static("─── PIPELINE STEPS ───", classes="panel-title")
                    for i, step_name in enumerate(StatusManager.STEPS):
                        yield PipelineStep(step_name, i, classes="pipeline-step")
                    
                    # Progress section
                    yield Static("", id="progress-label", classes="progress-label")
                    yield ProgressBar(id="progress-bar", total=100, show_eta=False)
                    yield Static("", id="progress-stats", classes="progress-stats")
                
                # Right panel - Logs
                with Vertical(id="pipeline-logs-panel", classes="pipeline-logs-panel"):
                    yield Static("─── STEP LOGS ───", classes="panel-title")
                    yield RichLog(id="log-viewer", highlight=True, markup=True, wrap=True)
            
            # Footer status bar
            with Horizontal(id="pipeline-footer", classes="pipeline-footer"):
                yield Static("Mode: Local | Status: Idle", id="footer-status")
                yield Static("Hints: ^C Quit, Tab Switch Focus", id="footer-hints")
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        # Set the status file based on the loaded config
        if self.filepath:
            self._current_status_file = StatusManager.get_status_file_path(self.filepath)
        self._poll_timer = self.set_interval(self.POLL_INTERVAL, self._poll_status)
    
    def on_unmount(self) -> None:
        """Called when the widget is unmounted."""
        if self._poll_timer:
            self._poll_timer.stop()
    
    def _poll_status(self) -> None:
        """Poll the current status file for updates."""
        if self._current_status_file:
            self._update_ui_from_status()
    
    def _update_ui_from_status(self) -> None:
        """Update the UI based on the current status file."""
        if not self._current_status_file:
            self._reset_ui()
            return
        
        status = StatusManager.read_status(self._current_status_file)
        if not status:
            self._reset_ui()
            return
        
        # Check if this is a new run (run_id changed)
        current_run_id = status.get("run_id")
        if current_run_id != self._last_run_id:
            self._last_run_id = current_run_id
            self._displayed_log_count = 0  # Reset log counter for new run
            # Clear logs for new run
            log_viewer = self.query_one("#log-viewer", RichLog)
            log_viewer.clear()
        
        # Update steps
        current_step_index = status.get("current_step_index", 0)
        pipeline_status = status.get("status", "initializing")
        
        steps = self.query(PipelineStep)
        for step in steps:
            if step.step_index < current_step_index:
                step.set_status("done")
            elif step.step_index == current_step_index:
                if pipeline_status == "completed":
                    step.set_status("done")
                elif pipeline_status == "error":
                    step.set_status("error")
                else:
                    step.set_status("running")
            else:
                step.set_status("pending")
        
        # Update progress
        progress = status.get("progress", {})
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        
        progress_label = self.query_one("#progress-label", Static)
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_stats = self.query_one("#progress-stats", Static)
        
        if total > 0:
            pct = int((completed + failed) / total * 100)
            progress_label.update(f"Progress: {completed + failed}/{total}")
            progress_bar.update(progress=pct)
            progress_stats.update(f"✓ {completed} completed | ✗ {failed} failed")
        else:
            progress_label.update("Progress: Waiting...")
            progress_bar.update(progress=0)
            progress_stats.update("")
        
        # Update logs - only add new entries since last update
        log_viewer = self.query_one("#log-viewer", RichLog)
        logs = status.get("logs", [])
        
        # Add any new log entries
        for log_entry in logs[self._displayed_log_count:]:
            # Color code based on content
            if "[WARNING]" in log_entry or "FAILED" in log_entry:
                log_viewer.write(f"[yellow]{log_entry}[/yellow]")
            elif "[ERROR]" in log_entry or "error" in log_entry.lower():
                log_viewer.write(f"[red]{log_entry}[/red]")
            elif "SUCCESS" in log_entry or "completed" in log_entry.lower():
                log_viewer.write(f"[green]{log_entry}[/green]")
            else:
                log_viewer.write(log_entry)
        
        # Update the count of displayed logs
        self._displayed_log_count = len(logs)
        
        # Check heartbeat - is the pipeline actually running?
        last_updated_str = status.get("last_updated")
        age = None
        self._is_pipeline_alive = False
        
        if last_updated_str:
            try:
                last_updated = datetime.fromisoformat(last_updated_str)
                age = (datetime.now() - last_updated).total_seconds()
                # Consider alive if status is running AND updated within timeout
                if pipeline_status == "running" and age < self.HEARTBEAT_TIMEOUT:
                    self._is_pipeline_alive = True
            except (ValueError, TypeError):
                pass
        
        # Update alive indicator
        indicator = self.query_one("#alive-indicator", Static)
        indicator.remove_class("alive-running", "alive-dead", "alive-completed", "alive-stale")
        
        if pipeline_status == "completed":
            indicator.add_class("alive-completed")
        elif pipeline_status == "error":
            indicator.add_class("alive-dead")
        elif pipeline_status == "running":
            # Show green if alive (recent heartbeat), yellow-ish if stale but still "running"
            if self._is_pipeline_alive:
                indicator.add_class("alive-running")
            else:
                indicator.add_class("alive-stale")  # New class for stale but running
        else:
            indicator.add_class("alive-dead")
        
        # Update footer with more info
        footer_status = self.query_one("#footer-status", Static)
        
        # Detect mode from config path in status file
        config_path = status.get("config_path", "")
        if "/content/" in config_path or "\\content\\" in config_path:
            mode = "Colab"
        elif "Google Drive" in config_path or "My Drive" in config_path:
            mode = "Drive"
        else:
            mode = "Local"
        
        # Show last update time for debugging sync issues
        age_str = ""
        if last_updated_str:
            try:
                last_updated = datetime.fromisoformat(last_updated_str)
                age = (datetime.now() - last_updated).total_seconds()
                if age < 60:
                    age_str = f" | Updated {int(age)}s ago"
                else:
                    age_str = f" | Updated {int(age/60)}m ago"
            except (ValueError, TypeError):
                pass
        
        status_text = pipeline_status.capitalize()
        alive_text = "LIVE" if self._is_pipeline_alive else ("Done" if pipeline_status == "completed" else "Stale")
        footer_status.update(f"Mode: {mode} | Status: {status_text} | {alive_text}{age_str}")
    
    def _reset_ui(self) -> None:
        """Reset the UI to idle state."""
        # Reset steps
        steps = self.query(PipelineStep)
        for step in steps:
            step.set_status("pending")
        
        # Reset progress
        self.query_one("#progress-label", Static).update("Progress: No data")
        self.query_one("#progress-bar", ProgressBar).update(progress=0)
        self.query_one("#progress-stats", Static).update("")
        
        # Clear logs
        self.query_one("#log-viewer", RichLog).clear()
        
        # Reset footer
        self.query_one("#footer-status", Static).update("Mode: -- | Status: Idle | No status file found")
