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
        ("Scale", "segmentation_options.rescaling_config.scale_factor", "float", None),
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
        # MPP validation state: {image_id: {"mpp_x": "correct|incorrect|unvalidated", "mpp_y": "..."}}
        self._mpp_validation_state: dict[str, dict[str, str]] = {}
        # Scale validation state: {image_id: {"scale_factor": "valid|invalid|unvalidated", "suggested_scale": float}}
        self._scale_validation_state: dict[str, dict[str, str | float]] = {}
        # File existence state: {image_id: bool} - True if file exists, False if not
        self._file_existence_state: dict[str, bool] = {}
        # Suggested maximum scaled dimension (in pixels)
        self.SUGGESTED_MAX_SCALED_SIZE = 8000
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the images view."""
        with Container(classes="images-container"):
            yield Static("Image Configurations", classes="images-title")
            
            yield DataTable(id="images-table", classes="images-table")

            with Horizontal(classes="images-toolbar"):
                yield Button("🔍 MPP", id="check-mpp",    classes="toolbar-button", variant="primary")
                yield Button("🧮 Scale", id="calc-scale",    classes="toolbar-button", variant="primary")
                yield Static("", classes="toolbar-spacer")
                yield Button("Add Images", id="add-image",    classes="toolbar-button", variant="primary")
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
        
        # Check if we have valid images (not just the default empty one)
        has_valid_images = False
        for img in self.config.image_configurations:
            if img.image_id != "default" and img.original_image_filename:
                has_valid_images = True
                break
        
        # Enable/disable buttons
        try:
            self.query_one("#check-mpp", Button).disabled = not has_valid_images
            self.query_one("#calc-scale", Button).disabled = not has_valid_images
        except Exception:
            pass # Buttons might not be mounted yet
        
        # Check file existence for all images
        import os
        for img_config in self.config.image_configurations:
            if img_config.original_image_filename:
                filepath = img_config.original_image_filename.replace("\\", "/")
                self._file_existence_state[img_config.image_id] = os.path.exists(filepath)
            else:
                self._file_existence_state[img_config.image_id] = False
        
        for img_config in self.config.image_configurations:
            row_data = []
            for col_name, field_path, value_type, options in self.COLUMNS:
                value = self._get_field_value(img_config, field_path)
                display_value = self._format_value(value, value_type, col_name, field_path, img_config)
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
    
    def _make_path_relative(self, filepath: str) -> str:
        """Convert an absolute path to a relative path (relative to PROJECT_ROOT)."""
        import os
        try:
            from src.file_paths import PROJECT_ROOT
            
            # Normalize paths
            abs_path = os.path.abspath(filepath).replace('\\', '/')
            project_root = os.path.abspath(PROJECT_ROOT).replace('\\', '/')
            
            # Check if the path is within the project root
            if abs_path.startswith(project_root):
                # Get relative path
                relative = os.path.relpath(abs_path, project_root)
                return relative.replace('\\', '/')
            else:
                # Path is outside project root - return as-is but log a warning
                # In this case, we can't make it relative, so we'll store it as absolute
                # but this might cause issues on different platforms
                return filepath
        except (ImportError, Exception):
            # If we can't determine PROJECT_ROOT, return as-is
            return filepath
    
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
                        attr_value = TilingParameters(apply_tiling=False)
                        setattr(obj, part, attr_value)
                    elif part == "segmentation_options":
                        from tui.models import SegmentationOptions
                        attr_value = SegmentationOptions()
                        setattr(obj, part, attr_value)
                obj = attr_value
            else:
                return
        # Convert absolute paths to relative paths for original_image_filename before setting
        if field_path == "original_image_filename" and isinstance(value, str) and value:
            value = self._make_path_relative(value)
        
        # Set the final attribute
        setattr(obj, parts[-1], value)
        
        # Reset validation state if MPP values are changed
        if field_path in ("mpp_x", "mpp_y") and hasattr(img_config, 'image_id'):
            image_id = img_config.image_id
            if image_id in self._mpp_validation_state:
                self._mpp_validation_state[image_id][field_path] = "unvalidated"
        
        # Reset validation state if scale factor is changed
        # Check if field_path ends with "scale_factor" to handle nested paths
        if field_path.endswith("scale_factor") and hasattr(img_config, 'image_id'):
            image_id = img_config.image_id
            if image_id in self._scale_validation_state:
                self._scale_validation_state[image_id]["scale_factor"] = "unvalidated"
        
        # Check file existence if filename is changed
        if field_path == "original_image_filename" and hasattr(img_config, 'image_id'):
            image_id = img_config.image_id
            import os
            if value and isinstance(value, str):
                # Use resolve_image_path to handle both relative and absolute paths
                try:
                    from src.pipeline_utils import resolve_image_path
                    resolved_path = resolve_image_path(value)
                    self._file_existence_state[image_id] = os.path.exists(resolved_path)
                except ImportError:
                    # Fallback to simple check
                    filepath = value.replace("\\", "/")
                    self._file_existence_state[image_id] = os.path.exists(filepath)
            else:
                self._file_existence_state[image_id] = False
            # Refresh table to show updated file existence color
            self.refresh_table()
    
    def _format_value(self, value: Any, value_type: str, col_name: str = "", field_path: str = "", img_config: ImageConfiguration = None) -> str | Text:
        """Format a value for display in the table."""
        if value is None:
            return "N/A"
        
        # File existence check - use field_path for robustness
        if field_path == "original_image_filename" and isinstance(value, str) and value:
            # Shorten filename for display
            path = Path(value)
            display_name = path.name
            if len(path.parts) > 1:
                display_name = f".../{path.name}"
            
            # Check file existence and color accordingly
            if img_config and img_config.image_id in self._file_existence_state:
                file_exists = self._file_existence_state[img_config.image_id]
                if not file_exists:
                    return Text(display_name, style="red")
            
            return display_name
            
        elif value_type == "bool":
            if value:
                return Text("✓", style="green")
            else:
                return Text("✗", style="red")
        elif field_path in ("mpp_x", "mpp_y") and img_config:
            # Check validation state for MPP values using field_path
            image_id = img_config.image_id
            field_name = field_path  # Use field_path directly
            
            # Format MPP value to max 4 decimal places
            formatted_value = f"{float(value):.4f}".rstrip('0').rstrip('.')
            
            if image_id in self._mpp_validation_state:
                validation = self._mpp_validation_state[image_id].get(field_name, "unvalidated")
                if validation == "correct":
                    return Text(formatted_value, style="green")
                elif validation == "incorrect":
                    return Text(formatted_value, style="red")
            
            # Default: unvalidated or no validation state
            return formatted_value
        elif field_path.endswith("scale_factor") and img_config:
            # Check validation state for scale factors using field_path
            image_id = img_config.image_id
            
            # Format scale value to 1 decimal place
            formatted_value = f"{float(value):.1f}".rstrip('0').rstrip('.')
            
            if image_id in self._scale_validation_state:
                validation = self._scale_validation_state[image_id].get("scale_factor", "unvalidated")
                if validation == "valid":
                    return Text(formatted_value, style="green")
                elif validation == "invalid":
                    return Text(formatted_value, style="red")
            
            # Default: unvalidated or no validation state
            return formatted_value
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
    
    def check_all_mpp_values(self) -> None:
        """Check and validate MPP values for all images."""
        from tui.utils.mpp_calculator import calculate_mpp_from_image
        
        valid_images = [img for img in self.config.image_configurations 
                       if img.image_id != "default" and img.original_image_filename]
        
        if not valid_images:
            self.notify("No valid images to check", severity="warning")
            return
        
        self.notify(f"Checking MPP values for {len(valid_images)} image(s)...")
        
        updated_count = 0
        validated_count = 0
        error_count = 0
        
        for img_config in valid_images:
            filepath = img_config.original_image_filename
            image_id = img_config.image_id
            
            # Initialize validation state if not present
            if image_id not in self._mpp_validation_state:
                self._mpp_validation_state[image_id] = {"mpp_x": "unvalidated", "mpp_y": "unvalidated"}
            
            # Calculate MPP from image metadata
            calculated_mpp_x, calculated_mpp_y = calculate_mpp_from_image(filepath)
            
            if calculated_mpp_x is None and calculated_mpp_y is None:
                # Could not calculate MPP
                self._mpp_validation_state[image_id]["mpp_x"] = "unvalidated"
                self._mpp_validation_state[image_id]["mpp_y"] = "unvalidated"
                error_count += 1
                continue
            
            # Handle mpp_x
            if calculated_mpp_x is not None:
                if img_config.mpp_x is None:
                    # Auto-update missing value
                    img_config.mpp_x = calculated_mpp_x
                    self._mpp_validation_state[image_id]["mpp_x"] = "correct"
                    updated_count += 1
                else:
                    # Validate existing value
                    diff = abs(calculated_mpp_x - img_config.mpp_x)
                    if diff <= 0.001:
                        self._mpp_validation_state[image_id]["mpp_x"] = "correct"
                        validated_count += 1
                    else:
                        self._mpp_validation_state[image_id]["mpp_x"] = "incorrect"
            
            # Handle mpp_y
            if calculated_mpp_y is not None:
                if img_config.mpp_y is None:
                    # Auto-update missing value
                    img_config.mpp_y = calculated_mpp_y
                    self._mpp_validation_state[image_id]["mpp_y"] = "correct"
                    updated_count += 1
                else:
                    # Validate existing value
                    diff = abs(calculated_mpp_y - img_config.mpp_y)
                    if diff <= 0.001:
                        self._mpp_validation_state[image_id]["mpp_y"] = "correct"
                        validated_count += 1
                    else:
                        self._mpp_validation_state[image_id]["mpp_y"] = "incorrect"
        
        # Refresh table to show updated colors
        self.refresh_table()
        
        # Show summary notification
        if updated_count > 0:
            self.notify(f"Updated {updated_count} MPP value(s), validated {validated_count} value(s)")
        elif validated_count > 0:
            self.notify(f"Validated {validated_count} MPP value(s)")
        elif error_count > 0:
            self.notify(f"Could not calculate MPP for {error_count} image(s)", severity="warning")
        else:
            self.notify("MPP check complete")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "add-image":
            self.add_image()
        elif event.button.id == "remove-image":
            self.remove_image()
        elif event.button.id == "check-mpp":
            self.check_all_mpp_values()
        elif event.button.id == "calc-scale":
            self.check_all_scale_values()
    
    def add_image(self) -> None:
        """Add new image configuration(s)."""
        from tui.screens.image_picker import ImagePicker
        import os
        
        # Determine initial path
        # Default to current working directory or a sensible default
        initial_path = os.getcwd()
        try:
            from src.file_paths import PROJECT_ROOT
            if os.path.exists(PROJECT_ROOT):
                initial_path = PROJECT_ROOT
        except ImportError:
            pass
        
        def on_dismiss(results: list[str]) -> None:
            if not results:
                return
            
            # Remove default image if it exists before adding new ones
            if self.config.image_configurations:
                default_img = next((img for img in self.config.image_configurations if img.image_id == "default"), None)
                if default_img:
                    self.config.image_configurations.remove(default_img)
            
            count = 0
            for filepath in results:
                # Convert absolute path to relative path (relative to PROJECT_ROOT)
                relative_path = self._make_path_relative(filepath)
                
                # Create unique ID
                # Simple generation based on count, but ensuring uniqueness
                base_id = f"image_{len(self.config.image_configurations) + 1}"
                image_id = base_id
                
                # Check for duplicate ID
                existing_ids = {img.image_id for img in self.config.image_configurations}
                counter = 1
                while image_id in existing_ids:
                    image_id = f"{base_id}_{counter}"
                    counter += 1
                
                new_image = ImageConfiguration(
                    image_id=image_id,
                    original_image_filename=relative_path
                )
                self.config.image_configurations.append(new_image)
                count += 1
            
            if count > 0:
                self.refresh_table()
                self.notify(f"Added {count} image(s)")
        
        self.app.push_screen(ImagePicker(initial_path), on_dismiss)
    
    def check_all_scale_values(self) -> None:
        """Check and validate scale factors for all images."""
        from tui.utils.mpp_calculator import get_image_dimensions
        
        valid_images = [img for img in self.config.image_configurations 
                       if img.image_id != "default" and img.original_image_filename]
        
        if not valid_images:
            self.notify("No valid images to check", severity="warning")
            return
        
        self.notify(f"Checking scale factors for {len(valid_images)} image(s)...")
        
        checked_count = 0
        suggested_count = 0
        error_count = 0
        
        for img_config in valid_images:
            filepath = img_config.original_image_filename
            image_id = img_config.image_id
            
            # Initialize validation state if not present
            if image_id not in self._scale_validation_state:
                self._scale_validation_state[image_id] = {
                    "scale_factor": "unvalidated",
                    "suggested_scale": 1.0
                }
            
            # Get image dimensions
            width, height = get_image_dimensions(filepath)
            
            if width is None or height is None:
                self._scale_validation_state[image_id]["scale_factor"] = "unvalidated"
                error_count += 1
                continue
            
            # Get current scale factor
            rescaling_config = img_config.segmentation_options.rescaling_config
            current_scale = rescaling_config.scale_factor if rescaling_config else None
            
            # Calculate suggested scale factor
            # Use the larger dimension to determine the scale
            larger_dimension = max(width, height)
            if larger_dimension > self.SUGGESTED_MAX_SCALED_SIZE:
                suggested_scale = self.SUGGESTED_MAX_SCALED_SIZE / larger_dimension
                # Round to 4 decimal places for cleaner display
                suggested_scale = round(suggested_scale, 4)
                # Ensure it's within valid range (0 < scale <= 1.0)
                suggested_scale = max(0.0001, min(1.0, suggested_scale))
            else:
                # Image is already small enough, suggest 1.0
                suggested_scale = 1.0
            
            self._scale_validation_state[image_id]["suggested_scale"] = suggested_scale
            
            # Auto-fill suggested scale if current scale is empty/default/NA
            # (i.e., rescaling_config is None, or scale_factor is None, or scale_factor is default 1.0)
            should_auto_fill = False
            if rescaling_config is None:
                # No rescaling config exists, create it with suggested scale
                from tui.models import RescalingConfig
                img_config.segmentation_options.rescaling_config = RescalingConfig(scale_factor=suggested_scale)
                should_auto_fill = True
            elif current_scale is None:
                # Scale is None, set to suggested
                rescaling_config.scale_factor = suggested_scale
                should_auto_fill = True
            elif current_scale == 1.0:
                # Scale is default 1.0, update to suggested (even if suggested is also 1.0, to ensure it's set)
                rescaling_config.scale_factor = suggested_scale
                if suggested_scale != 1.0:
                    should_auto_fill = True
            
            # Use the (potentially updated) scale for validation
            final_scale = rescaling_config.scale_factor if rescaling_config else suggested_scale
            
            # Calculate scaled dimensions
            scaled_width = int(width * final_scale)
            scaled_height = int(height * final_scale)
            
            # Check if scaled dimensions exceed maximum
            max_dimension = max(scaled_width, scaled_height)
            is_valid = max_dimension <= self.SUGGESTED_MAX_SCALED_SIZE
            
            # Update validation state
            self._scale_validation_state[image_id]["scale_factor"] = "valid" if is_valid else "invalid"
            
            checked_count += 1
            if should_auto_fill:
                suggested_count += 1
            elif suggested_scale != final_scale and suggested_scale < 1.0:
                suggested_count += 1
        
        # Refresh table to show updated colors
        self.refresh_table()
        
        # Show summary notification with suggestions
        if suggested_count > 0:
            self.notify(f"Checked {checked_count} image(s). {suggested_count} may benefit from scale adjustment. Check table for suggestions.", severity="info")
        elif checked_count > 0:
            self.notify(f"Checked {checked_count} image(s). All scale factors are appropriate.")
        elif error_count > 0:
            self.notify(f"Could not read dimensions for {error_count} image(s)", severity="warning")
        else:
            self.notify("Scale check complete")
    
    def remove_image(self) -> None:
        """Remove the selected image."""
        table = self.query_one("#images-table", DataTable)
        cursor_row = table.cursor_row
        
        if cursor_row is None or cursor_row >= len(self.config.image_configurations):
            self.notify("Please select an image to remove", severity="warning")
            return
        
        # Don't allow removing the last image
        if len(self.config.image_configurations) <= 1:
            self.notify("Cannot remove the last image", severity="warning")
            return
        
        image_config = self.config.image_configurations[cursor_row]
        image_id = image_config.image_id
        self.config.image_configurations.remove(image_config)
        # Clean up validation state
        if image_id in self._mpp_validation_state:
            del self._mpp_validation_state[image_id]
        if image_id in self._scale_validation_state:
            del self._scale_validation_state[image_id]
        if image_id in self._file_existence_state:
            del self._file_existence_state[image_id]
        self.refresh_table()
        self.notify(f"Removed {image_config.image_id}")
