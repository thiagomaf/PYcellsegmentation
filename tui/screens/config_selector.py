"""Modal screen for selecting a config file to remove."""
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, RadioButton, RadioSet

from tui.optimization.models import ConfigFileInfo


class ConfigSelector(ModalScreen[str]):
    """Modal screen for selecting a config file."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, config_files: list[ConfigFileInfo], title: str = "Select Config"):
        """Initialize the config selector.
        
        Args:
            config_files: List of ConfigFileInfo objects to choose from
            title: Title to display
        """
        super().__init__()
        self.config_files = config_files
        self.title = title
        self.selected_filepath = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the selector."""
        with Container(classes="config-selector-container"):
            yield Static(self.title, classes="config-selector-header")
            
            with Vertical(classes="config-selector-list"):
                with RadioSet(id="config-radio-set"):
                    for config_info in self.config_files:
                        config_name = Path(config_info.filepath).name
                        yield RadioButton(config_name, value=False)
            
            with Horizontal(classes="config-selector-footer"):
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Remove", id="remove", variant="error")
    
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio button selection."""
        # Use pressed_index to get the selected button index
        radio_set = event.radio_set
        pressed_index = radio_set.pressed_index
        if pressed_index is not None and pressed_index < len(self.config_files):
            self.selected_filepath = self.config_files[pressed_index].filepath
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "remove":
            # Check if a radio button is selected
            radio_set = self.query_one("#config-radio-set", RadioSet)
            pressed_button = radio_set.pressed_button
            if pressed_button:
                # Find the index of the selected button
                radio_buttons = list(radio_set.query(RadioButton))
                try:
                    selected_index = radio_buttons.index(pressed_button)
                    if selected_index < len(self.config_files):
                        selected_filepath = self.config_files[selected_index].filepath
                        self.dismiss(selected_filepath)
                    else:
                        self.notify("Invalid selection", severity="error")
                except (ValueError, IndexError):
                    self.notify("Please select a config file to remove", severity="warning")
            else:
                self.notify("Please select a config file to remove", severity="warning")

