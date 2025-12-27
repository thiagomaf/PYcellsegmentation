"""Dashboard screen for the TUI."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Button, Static

ascii_art = """


 ███████████  █████ █████                   ████  ████                           
░░███░░░░░███░░███ ░░███                   ░░███ ░░███                           
 ░███    ░███ ░░███ ███    ██████   ██████  ░███  ░███   █████   ██████   ███████
 ░██████████   ░░█████    ███░░███ ███░░███ ░███  ░███  ███░░   ███░░███ ███░░███
 ░███░░░░░░     ░░███    ░███ ░░░ ░███████  ░███  ░███ ░░█████ ░███████ ░███ ░███
 ░███            ░███    ░███  ███░███░░░   ░███  ░███  ░░░░███░███░░░  ░███ ░███
 █████           █████   ░░██████ ░░██████  █████ █████ ██████ ░░██████ ░░███████
░░░░░           ░░░░░     ░░░░░░   ░░░░░░  ░░░░░ ░░░░░ ░░░░░░   ░░░░░░   ░░░░░███
                                                                         ███ ░███
                                                                        ░░██████ 
                                                                         ░░░░░░  
"""


class Dashboard(Container):
    """Main dashboard screen for project management."""
    
    BINDINGS = [
        Binding("n", "new_project",  "New Project",  tooltip="Create a new project"),
        Binding("l", "load_project", "Load Project", tooltip="Load a project"),
    ]
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the dashboard."""
        TITLE_ART = ascii_art

        with Container(classes="dashboard-content"):
            yield Static(TITLE_ART, classes="dashboard-title", markup=False)
            yield Static("Project Manager", classes="dashboard-subtitle")
            
            with Vertical(classes="dashboard-buttons"):
                yield Button("New Project",  id="new-project",  classes="dashboard-button")
                yield Button("Load Project", id="load-project", classes="dashboard-button")
                yield Button("Exit",         id="exit",         classes="dashboard-button")
    
    def on_mount(self) -> None:
        pass
