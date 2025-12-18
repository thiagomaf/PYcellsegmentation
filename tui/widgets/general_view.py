"""General settings view widget."""
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Input, Select, Static, Switch
from textual.reactive import reactive

from tui.models import ProjectConfig, GlobalSettings


class GeneralView(Container):
    """View for editing global segmentation settings."""
    
    def __init__(self, config: ProjectConfig):
        """Initialize the general view."""
        super().__init__()
        self.config = config
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the general view."""
        settings = self.config.global_segmentation_settings
        
        with Vertical():
            with Container(classes="general-container"):
                yield Static("Global Segmentation Settings", classes="general-title")
                
                with Container(classes="form-row"):
                    yield Static("Log Level:", classes="form-label")
                    yield Select(
                        [
                            ("DEBUG", "DEBUG"),
                            ("INFO", "INFO"),
                            ("WARNING", "WARNING"),
                            ("ERROR", "ERROR"),
                        ],
                        value=settings.default_log_level,
                        id="log-level",
                        classes="form-input"
                    )
                
                with Container(classes="form-row"):
                    yield Static("Max Processes:", classes="form-label")
                    yield Input(
                        str(settings.max_processes),
                        id="max-processes",
                        classes="form-input",
                        type="integer"
                    )
                
                with Container(classes="form-row"):
                    yield Static("Force Grayscale:", classes="form-label")
                    yield Switch(
                        settings.FORCE_GRAYSCALE,
                        id="force-grayscale"
                    )
                
                with Container(classes="form-row"):
                    yield Static("Use GPU if Available:", classes="form-label")
                    yield Switch(
                        settings.USE_GPU_IF_AVAILABLE,
                        id="use-gpu"
                    )
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id == "max-processes":
            try:
                value = int(event.value)
                if value > 0:
                    self.config.global_segmentation_settings.max_processes = value
            except ValueError:
                pass
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "log-level":
            self.config.global_segmentation_settings.default_log_level = event.value
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        if event.switch.id == "force-grayscale":
            self.config.global_segmentation_settings.FORCE_GRAYSCALE = event.value
        elif event.switch.id == "use-gpu":
            self.config.global_segmentation_settings.USE_GPU_IF_AVAILABLE = event.value
