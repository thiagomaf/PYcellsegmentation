"""Cell editor widget for inline table editing."""
from pathlib import Path
from typing import Optional, Callable, Any
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static, Switch
from textual import events


class CellEditor(ModalScreen[Optional[Any]]):
    """Modal screen for editing a single table cell."""
    
    # CSS_PATH = str(Path(__file__).parent / "cell_editor.tcss")
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "save", "Save"),
    ]
    
    def __init__(
        self,
        column_name: str,
        value: Any,
        value_type: str,
        options: Optional[list] = None,
        validator: Optional[Callable[[str], Any]] = None,
        inline: bool = False
    ):
        """Initialize the cell editor.
        
        Args:
            column_name: Name of the column being edited
            value: Current value
            value_type: Type of value - "bool", "enum", "int", "float", "str"
            options: For enum types, list of (value, label) tuples
            validator: Optional function to validate and convert input
            inline: If True, create a compact inline editor
        """
        super().__init__()
        self.column_name = column_name
        self.value = value
        self.value_type = value_type
        self.options = options or []
        self.validator = validator
        self.result = None
        self.inline = inline
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the cell editor."""
        container_class = "cell-editor-container-inline" if self.inline else "cell-editor-container"
        
        with Container(classes=container_class):
            if not self.inline:
                yield Static(f"Edit {self.column_name}", classes="cell-editor-title")
            
            with Container(classes="cell-editor-content"):
                if self.value_type == "bool":
                    yield Switch(
                        bool(self.value) if self.value is not None else False,
                        id="cell-value"
                    )
                elif self.value_type == "enum":
                    current_value = str(self.value) if self.value is not None else ""
                    yield Select(
                        self.options,
                        value=current_value,
                        id="cell-value",
                        allow_blank=False
                    )
                else:
                    # int, float, str
                    display_value = str(self.value) if self.value is not None else ""
                    yield Input(
                        display_value,
                        id="cell-value",
                        classes="cell-editor-input"
                    )
            
            if not self.inline:
                with Horizontal(classes="cell-editor-footer"):
                    yield Button("Cancel", id="cancel", classes="footer-button")
                    yield Button("Save", id="save", classes="footer-button", variant="primary")
            else:
                # Inline mode: just show the input, Enter to save, Esc to cancel
                yield Static("Enter: Save | Esc: Cancel", classes="cell-editor-hint")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.action_cancel()
        elif event.button.id == "save":
            self.action_save()
    
    def on_mount(self) -> None:
        """Focus the input widget when mounted."""
        try:
            widget = self.query_one("#cell-value")
            if hasattr(widget, 'focus'):
                widget.focus()
        except Exception:
            pass
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in Input widget."""
        self.action_save()
    
    def action_cancel(self) -> None:
        """Cancel editing."""
        self.dismiss(None)
    
    def action_save(self) -> None:
        """Save the edited value."""
        try:
            widget = self.query_one("#cell-value")
            
            if self.value_type == "bool":
                if isinstance(widget, Switch):
                    result = widget.value
                else:
                    result = bool(self.value) if self.value is not None else False
            elif self.value_type == "enum":
                if isinstance(widget, Select):
                    result = widget.value
                    if result == "":
                        result = None
                else:
                    result = self.value
            elif self.value_type == "int":
                if isinstance(widget, Input):
                    value_str = widget.value.strip()
                    if not value_str:
                        result = None
                    else:
                        result = int(value_str)
                else:
                    result = self.value
            elif self.value_type == "float":
                if isinstance(widget, Input):
                    value_str = widget.value.strip()
                    if not value_str:
                        result = None
                    else:
                        result = float(value_str)
                else:
                    result = self.value
            else:  # str
                if isinstance(widget, Input):
                    value_str = widget.value.strip()
                    result = value_str if value_str else None
                else:
                    result = self.value
            
            # Apply validator if provided
            if self.validator:
                result = self.validator(result)
            
            self.dismiss(result)
        except ValueError as e:
            self.notify(f"Invalid value: {e}", severity="error")
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
