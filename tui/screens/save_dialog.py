"""Save dialog modal screen."""
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tui.models import ProjectConfig


class SaveDialog(ModalScreen[str]):
    """Modal screen for saving a project."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "save", "Save"),
    ]
    
    def __init__(self, config: ProjectConfig, initial_path: str = None):
        """Initialize the save dialog."""
        super().__init__()
        self.config = config
        if initial_path is None:
            # Default to config directory
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
            if config_dir.exists():
                self.initial_path = str(config_dir / "processing_config.json")
            else:
                self.initial_path = str(project_root / "processing_config.json")
        else:
            self.initial_path = initial_path
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the save dialog."""
        with Container(classes="file-picker-container"):
            yield Static("Save Project", classes="file-picker-header")
            
            yield Input(
                value=self.initial_path,
                placeholder="Path to save configuration file...",
                id="filepath-input",
                classes="file-picker-path"
            )
            
            with Horizontal(classes="file-picker-footer"):
                yield Button("Cancel", id="cancel", classes="file-picker-button")
                yield Button("Save", id="save", classes="file-picker-button", variant="primary")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.action_cancel()
        elif event.button.id == "save":
            self.action_save()
    
    def action_cancel(self) -> None:
        """Cancel saving."""
        self.dismiss(None)
    
    def action_save(self) -> None:
        """Save the project."""
        filepath = self.query_one("#filepath-input", Input).value
        
        if not filepath:
            self.notify("Please provide a file path", severity="error")
            return
        
        # Ensure .json extension
        if not filepath.endswith(".json"):
            filepath += ".json"
        
        try:
            # Create directory if it doesn't exist
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the config
            self.config.to_json_file(filepath)
            self.dismiss(filepath)
            self.app.notify(f"Project saved to {filepath}", severity="success")
        except Exception as e:
            self.notify(f"Error saving project: {e}", severity="error")

