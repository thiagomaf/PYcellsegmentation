"""JSON preview view widget."""
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import Static
from rich.syntax import Syntax

from tui.models import ProjectConfig


class PreviewView(Container):
    """View for previewing the JSON configuration."""
    
    def __init__(self, config: ProjectConfig):
        """Initialize the preview view."""
        super().__init__()
        self.config = config
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the preview view."""
        with Container(classes="preview-container"):
            yield Static("JSON Configuration Preview", classes="preview-title")
            with ScrollableContainer(id="json-preview"):
                yield Static("", id="json-content")
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.refresh_preview()
    
    def on_show(self) -> None:
        """Called when the widget is shown."""
        self.refresh_preview()
    
    def refresh_preview(self) -> None:
        """Refresh the JSON preview with syntax highlighting."""
        content = self.query_one("#json-content", Static)
        json_str = self.config.to_json()
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
        content.update(syntax)
