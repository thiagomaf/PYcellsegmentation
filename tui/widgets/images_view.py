"""Images management view widget."""

from pathlib import Path

from typing import Any
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, Static, Input, Switch, Select, Collapsible
from rich.text import Text
from textual import events

from tui.models import ProjectConfig, ImageConfiguration
from tui.widgets.cell_editor import CellEditor


class ImagesView(Container):
    """View for managing image configurations."""
    
    # CSS_PATH = str(Path(__file__).parent / ".images_view.tcss")
    BINDINGS = [
        Binding("space", "toggle_boolean", "Toggle boolean", tooltip="Toggle the boolean value of the selected cell"),
        Binding("enter", "edit_cell", "Edit cell (double click)", tooltip="Edit the selected cell inline"),
        Binding("ctrl+c", "copy_cell", "Copy", tooltip="Copy the selected cell value"),
        Binding("ctrl+v", "paste_cell", "Paste", tooltip="Paste value into the selected cell"),
        Binding("p", "paste_cell", "Paste", tooltip="Paste value into the selected cell"),
        # Block upstream navigation bindings from ConfigView and ProjectEditor
        Binding("1", "pass", show=False),
        Binding("2", "pass", show=False),
        Binding("3", "pass", show=False),
        Binding("4", "pass", show=False),
        Binding("c", "pass", show=False),
        Binding("r", "pass", show=False),
    ]
    
    def action_pass(self) -> None:
        """Do nothing to block upstream bindings."""
        pass

    # Column definitions: (name, field_path, type, options)
    COLUMNS = [
        ("ID", "image_id", "str", None),
        ("Filename", "original_image_filename", "str", None),
        ("Active", "is_active", "bool", None),
        ("MPP X", "mpp_x", "float", None),
        ("MPP Y", "mpp_y", "float", None),
        ("Segmentation", "segmentation_options.apply_segmentation", "bool", None),
        ("Scale Factor", "segmentation_options.rescaling_config.scale_factor", "float", None),
        ("Interpolation", "segmentation_options.rescaling_config.interpolation", "enum", [
            ("INTER_NEAREST", "INTER_NEAREST"),
            ("INTER_LINEAR", "INTER_LINEAR"),
            ("INTER_AREA", "INTER_AREA"),
            ("INTER_CUBIC", "INTER_CUBIC"),
            ("INTER_LANCZOS4", "INTER_LANCZOS4"),
        ]),
        ("Tiling", "segmentation_options.tiling_parameters.apply_tiling", "bool", None),
        ("Tile Size", "segmentation_options.tiling_parameters.tile_size", "int", None),
        ("Overlap", "segmentation_options.tiling_parameters.overlap", "int", None),
    ]
    
    def __init__(self, config: ProjectConfig):
        """Initialize the images view."""
        super().__init__()
        self.config = config
        self._clipboard = None
        self._last_click_time = {}
        self._double_click_threshold = 0.5  # seconds
        self.can_focus = True
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the images view."""
        with Vertical():
            with Container(classes="images-container"):
                yield Static("Image Configurations", classes="images-title")
                
                yield DataTable(id="images-table", classes="images-table")

                with Horizontal(classes="images-toolbar"):
                    yield Static("Enter: Edit | Double-click: Edit/Toggle | Space: Toggle boolean", id="status")
                    yield Static("", classes="toolbar-spacer")
                    yield Button("Add Image", id="add-image",    classes="toolbar-button", variant="primary")
                    yield Button("Remove",    id="remove-image", classes="toolbar-button")
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        # Focus this container so its bindings appear in the footer
        self.focus()
        table = self.query_one("#images-table", DataTable)
        # Add all columns
        column_names = [col[0] for col in self.COLUMNS]
        table.add_columns(*column_names)
        self.refresh_table()
    
    def refresh_table(self) -> None:
        """Refresh the images table."""
        table = self.query_one("#images-table", DataTable)
        table.clear()
        
        # Ensure we have at least one image configuration
        if not self.config.image_configurations:
            # Create a default one
            default_image = ImageConfiguration(
                image_id="default",
                original_image_filename=""
            )
            self.config.image_configurations.append(default_image)
        
        for img_config in self.config.image_configurations:
            row_data = []
            for col_name, field_path, value_type, options in self.COLUMNS:
                value = self._get_field_value(img_config, field_path)
                display_value = self._format_value(value, value_type, col_name)
                row_data.append(display_value)
            
            table.add_row(*row_data, key=img_config.image_id)
    
    def _get_field_value(self, img_config: ImageConfiguration, field_path: str) -> Any:
        """Get a field value from an image configuration using dot notation."""
        parts = field_path.split(".")
        obj = img_config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
                if obj is None:
                    return None
            else:
                return None
        return obj
    
    def _set_field_value(self, img_config: ImageConfiguration, field_path: str, value: Any) -> None:
        """Set a field value in an image configuration using dot notation."""
        from tui.models import RescalingConfig, TilingParameters
        
        parts = field_path.split(".")
        obj = img_config
        
        # Navigate to the parent object, creating nested objects if needed
        for i, part in enumerate(parts[:-1]):
            if hasattr(obj, part):
                attr_value = getattr(obj, part)
                if attr_value is None:
                    # Create the nested object if it doesn't exist
                    if part == "rescaling_config":
                        attr_value = RescalingConfig()
                        setattr(obj, part, attr_value)
                    elif part == "tiling_parameters":
                        attr_value = TilingParameters()
                        setattr(obj, part, attr_value)
                    elif part == "segmentation_options":
                        from tui.models import SegmentationOptions
                        attr_value = SegmentationOptions()
                        setattr(obj, part, attr_value)
                obj = attr_value
            else:
                return
        # Set the final attribute
        setattr(obj, parts[-1], value)
    
    def _format_value(self, value: Any, value_type: str, col_name: str = "") -> str | Text:
        """Format a value for display in the table."""
        if value is None:
            return "N/A"
        
        if col_name == "Filename" and isinstance(value, str) and value:
            # Shorten filename for display
            path = Path(value)
            # If there are directory parts, show .../filename
            if len(path.parts) > 1:
                return f".../{path.name}"
            return path.name
            
        elif value_type == "bool":
            if value:
                return Text("✓", style="green")
            else:
                return Text("✗", style="red")
        else:
            return str(value)
    
    def _convert_value(self, value: Any, target_type: str, col_def: tuple) -> Any:
        """Convert value to target type."""
        if target_type == "bool":
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)
        elif target_type == "int":
            return int(value)
        elif target_type == "float":
            return float(value)
        elif target_type == "str":
            return str(value)
        elif target_type == "enum":
             options = [opt[0] for opt in col_def[3]]
             if str(value) not in options:
                 raise ValueError(f"Invalid enum value: {value}")
             return str(value)
        return value

    def action_copy_cell(self) -> None:
        """Copy the selected cell value."""
        table = self.query_one("#images-table", DataTable)
        cursor_row = table.cursor_row
        cursor_column = table.cursor_column
        
        if cursor_row is None or cursor_column is None:
            return
            
        if cursor_row >= len(self.config.image_configurations):
            return
            
        img_config = self.config.image_configurations[cursor_row]
        col_def = self.COLUMNS[cursor_column]
        field_path = col_def[1]
        
        value = self._get_field_value(img_config, field_path)
        self._clipboard = value
        self.notify(f"Copied: {value}")

    def action_paste_cell(self) -> None:
        """Paste value into the selected cell(s)."""
        table = self.query_one("#images-table", DataTable)
        
        value = self._clipboard
        if value is None:
            self.notify("Clipboard is empty", severity="warning")
            return

        cells_to_paste = set()
        # Try to get multi-selection if available
        if hasattr(table, "selection") and table.selection:
             cells_to_paste = table.selection
        
        # Fallback to cursor if no selection or empty
        if not cells_to_paste:
            if table.cursor_coordinate:
                cells_to_paste.add(table.cursor_coordinate)
            else:
                self.notify("No cell selected", severity="warning")
                return

        count = 0
        error_count = 0
        
        for coord in cells_to_paste:
            row, col = coord.row, coord.column
            
            if row >= len(self.config.image_configurations):
                continue
                
            img_config = self.config.image_configurations[row]
            col_def = self.COLUMNS[col]
            field_path = col_def[1]
            target_type = col_def[2]
            
            try:
                new_value = self._convert_value(value, target_type, col_def)
                self._set_field_value(img_config, field_path, new_value)
                count += 1
            except (ValueError, TypeError) as e:
                error_count += 1
                # self.notify(f"Error pasting: {e}", severity="error") # Debug
        
        if count > 0:
            self.refresh_table()
            self.notify(f"Pasted: {value}")
        
        if error_count > 0:
            self.notify(f"Failed to paste into {error_count} cell(s)", severity="warning")

    async def on_key(self, event: events.Key) -> None:
        """Handle key presses for inline editing."""
        # Uncomment for debugging key codes
        # self.notify(f"Key: {event.key}", timeout=2)

        if event.key == "enter":
            self.edit_cell(inline=False)
            event.stop()
        elif event.key == "space":
            # Toggle boolean values
            table = self.query_one("#images-table", DataTable)
            cursor_row = table.cursor_row
            cursor_column = table.cursor_column
            
            if cursor_row is not None and cursor_column is not None:
                col_def = self.COLUMNS[cursor_column]
                if col_def[2] == "bool":  # value_type is bool
                    await self.toggle_boolean_cell(cursor_row, cursor_column)
                    event.stop()
        elif event.key == "ctrl+v":
            # Explicitly handle ctrl+v in case binding is missed
            self.action_paste_cell()
            event.stop()
    
    async def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Handle cell selection - detect double-click for inline editing or toggling booleans."""
        import time
        current_time = time.time()
        # Use row and column as the key for double-click detection
        click_key = (event.coordinate.row, event.coordinate.column)
        column_index = event.coordinate.column
        
        # Check if this is a double-click
        if click_key in self._last_click_time:
            time_diff = current_time - self._last_click_time[click_key]
            if time_diff < self._double_click_threshold:
                # Double-click detected!
                # Check if it's a boolean column - if so, toggle it
                if column_index < len(self.COLUMNS):
                    col_def = self.COLUMNS[column_index]
                    if col_def[2] == "bool":  # value_type is bool
                        # Toggle the boolean value
                        await self.toggle_boolean_cell(event.coordinate.row, column_index)
                    else:
                        # Open inline editor for non-boolean values
                        self.edit_cell(inline=True)
                else:
                    # Fallback to inline editor
                    self.edit_cell(inline=True)
                self._last_click_time.pop(click_key, None)
                return
        
        # Store click time for double-click detection
        self._last_click_time[click_key] = current_time
        
        # Clean up old entries
        self._last_click_time = {
            k: v for k, v in self._last_click_time.items()
            if current_time - v < self._double_click_threshold
        }
    
    async def toggle_boolean_cell(self, row: int, col: int) -> None:
        """Toggle a boolean cell value."""
        if row >= len(self.config.image_configurations):
            return
        
        img_config = self.config.image_configurations[row]
        col_def = self.COLUMNS[col]
        field_path = col_def[1]
        
        current_value = self._get_field_value(img_config, field_path)
        new_value = not bool(current_value) if current_value is not None else True
        self._set_field_value(img_config, field_path, new_value)
        self.refresh_table()
    
    def edit_cell(self, inline: bool = False) -> None:
        """Edit the currently selected cell.
        
        Args:
            inline: If True, use compact inline editor positioned over the cell
        """
        table = self.query_one("#images-table", DataTable)
        cursor_row = table.cursor_row
        cursor_column = table.cursor_column
        
        if cursor_row is None or cursor_column is None:
            return
        
        if cursor_row >= len(self.config.image_configurations):
            return
        
        img_config = self.config.image_configurations[cursor_row]
        col_def = self.COLUMNS[cursor_column]
        col_name, field_path, value_type, options = col_def
        
        current_value = self._get_field_value(img_config, field_path)
        
        editor = CellEditor(
            column_name=col_name,
            value=current_value,
            value_type=value_type,
            options=options,
            inline=inline
        )
        
        # Capture references for the callback closure
        view = self
        config_ref = img_config
        field_ref = field_path
        
        def on_dismiss(result):
            view.app.notify(f"Edit result: {result}", timeout=3)  # Debug
            if result is not None:
                view._set_field_value(config_ref, field_ref, result)
                view.refresh_table()
        
        self.app.push_screen(editor, on_dismiss)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "add-image":
            self.add_image()
        elif event.button.id == "remove-image":
            self.remove_image()
    
    def add_image(self) -> None:
        """Add a new image configuration."""
        from tui.screens.image_editor import ImageEditor
        new_image = ImageConfiguration(
            image_id=f"image_{len(self.config.image_configurations) + 1}",
            original_image_filename=""
        )
        
        def on_dismiss(result):
            if result is not None:
                self.config.image_configurations.append(result)
                self.refresh_table()
        
        self.app.push_screen(ImageEditor(new_image), on_dismiss)
    
    def remove_image(self) -> None:
        """Remove the selected image."""
        table = self.query_one("#images-table", DataTable)
        cursor_row = table.cursor_row
        
        if cursor_row is None or cursor_row >= len(self.config.image_configurations):
            self.query_one("#status", Static).update("Please select an image to remove")
            return
        
        # Don't allow removing the last image
        if len(self.config.image_configurations) <= 1:
            self.query_one("#status", Static).update("Cannot remove the last image")
            return
        
        image_config = self.config.image_configurations[cursor_row]
        self.config.image_configurations.remove(image_config)
        self.refresh_table()
        self.query_one("#status", Static).update(f"Removed {image_config.image_id}")
