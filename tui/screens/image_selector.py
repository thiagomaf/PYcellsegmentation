"""Modal screen for selecting an image to remove."""
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Button, Static, RadioButton, RadioSet

from tui.optimization.models import ImagePoolEntry


class ImageSelector(ModalScreen[str]):
    """Modal screen for selecting an image."""
    
    CSS_PATH = str(Path(__file__).parent / "image_selector.tcss")
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, image_entries: list[ImagePoolEntry], title: str = "Select Image"):
        """Initialize the image selector.
        
        Args:
            image_entries: List of ImagePoolEntry objects to choose from
            title: Title to display
        """
        super().__init__()
        self.image_entries = image_entries
        self.title = title
        self.selected_filepath = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the selector."""
        with Container(classes="image-selector-container"):
            yield Static(self.title, classes="image-selector-header")
            
            with ScrollableContainer(classes="image-selector-list"):
                with RadioSet(id="image-radio-set"):
                    for image_entry in self.image_entries:
                        image_name = Path(image_entry.filepath).name
                        yield RadioButton(image_name, value=False)
            
            with Horizontal(classes="image-selector-footer"):
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Remove", id="remove", variant="error")
    
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio button selection."""
        # Use pressed_index to get the selected button index
        radio_set = event.radio_set
        pressed_index = radio_set.pressed_index
        if pressed_index is not None and pressed_index < len(self.image_entries):
            self.selected_filepath = self.image_entries[pressed_index].filepath
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "remove":
            # Check if a radio button is selected
            radio_set = self.query_one("#image-radio-set", RadioSet)
            pressed_button = radio_set.pressed_button
            if pressed_button:
                # Find the index of the selected button
                radio_buttons = list(radio_set.query(RadioButton))
                try:
                    selected_index = radio_buttons.index(pressed_button)
                    if selected_index < len(self.image_entries):
                        selected_filepath = self.image_entries[selected_index].filepath
                        self.dismiss(selected_filepath)
                    else:
                        self.notify("Invalid selection", severity="error")
                except (ValueError, IndexError):
                    self.notify("Please select an image to remove", severity="warning")
            else:
                self.notify("Please select an image to remove", severity="warning")

