from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static

from tui.models import ProjectConfig

class BaseView(Container):
    """Base class for ProjectEditor views."""
    
    def __init__(self, config: ProjectConfig, filepath: str | None = None):
        super().__init__()
        self.config = config
        self.filepath = filepath
    
    def compose(self) -> ComposeResult:
        """Override in subclasses to define view layout."""
        with Horizontal():
            # Upper section (control panel/summary)
            with Vertical(id="view-upper-section"):
                yield Static("Upper Section", id="view-summary")
            
            # Lower section (main content)
            with Vertical(id="view-lower-section"):
                yield Static("Lower Section", id="view-content")