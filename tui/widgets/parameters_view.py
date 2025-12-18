"""Parameters management view widget."""
from pathlib import Path
from typing import Any
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, Static
from rich.text import Text
from textual import events

from tui.models import ProjectConfig, ParameterConfiguration, CellposeParameters
from tui.widgets.cell_editor import CellEditor


class ParametersView(Container):
    """View for managing parameter configurations."""
    
    # CSS_PATH = str(Path(__file__).parent / "parameters_view.tcss")

    BINDINGS = [
        ("ctrl+c", "copy_cell", "Copy"),
        ("ctrl+v", "paste_cell", "Paste"),
        ("p", "paste_cell", "Paste"),
    ]
    
    # Column definitions: (name, field_path, type, options)
    COLUMNS = [
        ("ID", "param_set_id", "str", None),
        ("Active", "is_active", "bool", None),
        ("Model", "cellpose_parameters.MODEL_CHOICE", "enum", [
            ("cyto3", "cyto3"),
            ("nuclei", "nuclei"),
            ("cyto2", "cyto2"),
            ("cyto", "cyto"),
        ]),
        ("Diameter", "cellpose_parameters.DIAMETER", "int", None),
        ("Min Size", "cellpose_parameters.MIN_SIZE", "int", None),
        ("Cellprob", "cellpose_parameters.CELLPROB_THRESHOLD", "float", None),
        ("Force Grayscale", "cellpose_parameters.FORCE_GRAYSCALE", "bool", None),
        ("Z Projection", "cellpose_parameters.Z_PROJECTION_METHOD", "enum", [
            ("max", "max"),
            ("mean", "mean"),
            ("none", "none"),
        ]),
        ("Channel Index", "cellpose_parameters.CHANNEL_INDEX", "int", None),
        ("Enable 3D", "cellpose_parameters.ENABLE_3D_SEGMENTATION", "bool", None),
        ("Use GPU", "cellpose_parameters.USE_GPU", "bool", None),
    ]
    
    def __init__(self, config: ProjectConfig):
        """Initialize the parameters view."""
        super().__init__()
        self.config = config
        self._clipboard = None
        self._last_click_time = {}
        self._double_click_threshold = 0.5  # seconds
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the parameters view."""
        with Vertical():
            with Container(classes="parameters-container"):
                yield Static("Parameter Configurations", classes="parameters-title")                    
                
                yield DataTable(id="parameters-table", classes="parameters-table")

                with Horizontal(classes="parameters-toolbar"):
                    yield Static("Enter: Edit | Double-click: Edit/Toggle | Space: Toggle boolean", id="status")
                    yield Static("", classes="toolbar-spacer")
                    yield Button("Add Parameter Set", id="add-param", classes="toolbar-button", variant="primary")
                    yield Button("Remove", id="remove-param", classes="toolbar-button")
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        table = self.query_one("#parameters-table", DataTable)
        # Add all columns
        column_names = [col[0] for col in self.COLUMNS]
        table.add_columns(*column_names)
        self.refresh_table()
    
    def refresh_table(self) -> None:
        """Refresh the parameters table."""
        table = self.query_one("#parameters-table", DataTable)
        table.clear()
        
        # Ensure we have at least one parameter set
        if not self.config.cellpose_parameter_configurations:
            # Create a default one
            default_param = ParameterConfiguration(
                param_set_id="default",
                cellpose_parameters=CellposeParameters()
            )
            self.config.cellpose_parameter_configurations.append(default_param)
        
        for param_config in self.config.cellpose_parameter_configurations:
            row_data = []
            for col_name, field_path, value_type, options in self.COLUMNS:
                value = self._get_field_value(param_config, field_path)
                display_value = self._format_value(value, value_type)
                row_data.append(display_value)
            
            table.add_row(*row_data, key=param_config.param_set_id)
    
    def _get_field_value(self, param_config: ParameterConfiguration, field_path: str) -> Any:
        """Get a field value from a parameter configuration using dot notation."""
        parts = field_path.split(".")
        obj = param_config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj
    
    def _set_field_value(self, param_config: ParameterConfiguration, field_path: str, value: Any) -> None:
        """Set a field value in a parameter configuration using dot notation."""
        parts = field_path.split(".")
        obj = param_config
        # Navigate to the parent object
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return
        # Set the final attribute
        setattr(obj, parts[-1], value)
    
    def _format_value(self, value: Any, value_type: str) -> str | Text:
        """Format a value for display in the table."""
        if value is None:
            return "N/A"
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
        table = self.query_one("#parameters-table", DataTable)
        cursor_row = table.cursor_row
        cursor_column = table.cursor_column
        
        if cursor_row is None or cursor_column is None:
            return
            
        if cursor_row >= len(self.config.cellpose_parameter_configurations):
            return
            
        param_config = self.config.cellpose_parameter_configurations[cursor_row]
        col_def = self.COLUMNS[cursor_column]
        field_path = col_def[1]
        
        value = self._get_field_value(param_config, field_path)
        self._clipboard = value
        self.notify(f"Copied: {value}")

    def action_paste_cell(self) -> None:
        """Paste value into the selected cell(s)."""
        table = self.query_one("#parameters-table", DataTable)
        
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
            
            if row >= len(self.config.cellpose_parameter_configurations):
                continue
                
            param_config = self.config.cellpose_parameter_configurations[row]
            col_def = self.COLUMNS[col]
            field_path = col_def[1]
            target_type = col_def[2]
            
            try:
                new_value = self._convert_value(value, target_type, col_def)
                self._set_field_value(param_config, field_path, new_value)
                count += 1
            except (ValueError, TypeError):
                error_count += 1
        
        if count > 0:
            self.refresh_table()
            self.notify(f"Pasted: {value}")
        
        if error_count > 0:
            self.notify(f"Failed to paste into {error_count} cell(s)", severity="warning")
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        pass
    
    async def on_key(self, event: events.Key) -> None:
        """Handle key presses for inline editing."""
        if event.key == "enter":
            self.edit_cell(inline=False)
            event.stop()
        elif event.key == "space":
            # Toggle boolean values
            table = self.query_one("#parameters-table", DataTable)
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
    
    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Handle cell selection - detect double-click for inline editing or toggling booleans."""
        import time
        import asyncio
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
                        asyncio.create_task(self.toggle_boolean_cell(event.coordinate.row, column_index))
                    else:
                        # Open inline editor for non-boolean values
                        asyncio.create_task(self.edit_cell(inline=True))
                else:
                    # Fallback to inline editor
                    asyncio.create_task(self.edit_cell(inline=True))
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
        if row >= len(self.config.cellpose_parameter_configurations):
            return
        
        param_config = self.config.cellpose_parameter_configurations[row]
        col_def = self.COLUMNS[col]
        field_path = col_def[1]
        
        current_value = self._get_field_value(param_config, field_path)
        new_value = not bool(current_value) if current_value is not None else True
        self._set_field_value(param_config, field_path, new_value)
        self.refresh_table()
    
    def edit_cell(self, inline: bool = False) -> None:
        """Edit the currently selected cell.
        
        Args:
            inline: If True, use compact inline editor positioned over the cell
        """
        table = self.query_one("#parameters-table", DataTable)
        cursor_row = table.cursor_row
        cursor_column = table.cursor_column
        
        if cursor_row is None or cursor_column is None:
            return
        
        if cursor_row >= len(self.config.cellpose_parameter_configurations):
            return
        
        param_config = self.config.cellpose_parameter_configurations[cursor_row]
        col_def = self.COLUMNS[cursor_column]
        col_name, field_path, value_type, options = col_def
        
        current_value = self._get_field_value(param_config, field_path)
        
        editor = CellEditor(
            column_name=col_name,
            value=current_value,
            value_type=value_type,
            options=options,
            inline=inline
        )
        
        # Capture references for the callback closure
        view = self
        config_ref = param_config
        field_ref = field_path
        
        def on_dismiss(result):
            if result is not None:
                view._set_field_value(config_ref, field_ref, result)
                view.refresh_table()
        
        self.app.push_screen(editor, on_dismiss)
    
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "add-param":
            self.add_parameter()
        elif event.button.id == "remove-param":
            self.remove_parameter()
    
    def add_parameter(self) -> None:
        """Add a new parameter configuration."""
        from tui.screens.parameter_editor import ParameterEditor
        new_param = ParameterConfiguration(
            param_set_id=f"param_set_{len(self.config.cellpose_parameter_configurations) + 1}",
            cellpose_parameters=CellposeParameters()
        )
        
        def on_dismiss(result):
            if result is not None:
                self.config.cellpose_parameter_configurations.append(result)
                self.refresh_table()
        
        self.app.push_screen(ParameterEditor(new_param), on_dismiss)
    
    def remove_parameter(self) -> None:
        """Remove the selected parameter configuration."""
        table = self.query_one("#parameters-table", DataTable)
        cursor_row = table.cursor_row
        
        if cursor_row is None or cursor_row >= len(self.config.cellpose_parameter_configurations):
            self.query_one("#status", Static).update("Please select a parameter set to remove")
            return
        
        # Don't allow removing the last parameter set
        if len(self.config.cellpose_parameter_configurations) <= 1:
            self.query_one("#status", Static).update("Cannot remove the last parameter set")
            return
        
        param_config = self.config.cellpose_parameter_configurations[cursor_row]
        self.config.cellpose_parameter_configurations.remove(param_config)
        self.refresh_table()
        self.query_one("#status", Static).update(f"Removed {param_config.param_set_id}")
