"""File picker screen for loading projects."""
# import os
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Static
# from textual import events


class FilePicker(ModalScreen[str]):
    """Modal screen for picking a file."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "confirm", "Load"),
    ]
    
    def __init__(self, initial_path: str = None, allowed_extensions: set = None, title: str = "Select File"):
        """Initialize the file picker.
        
        Args:
            initial_path: Initial directory path to show
            allowed_extensions: Set of allowed file extensions (e.g., {'.json'}, {'.tif', '.tiff'})
            title: Title to display in the header
        """
        super().__init__()
        if initial_path is None:
            # Default to config directory or project root
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
            if config_dir.exists():
                self.initial_path = str(config_dir)
            else:
                self.initial_path = str(project_root)
        else:
            self.initial_path = initial_path
        self.allowed_extensions = allowed_extensions or {'.json'}
        self.title = title
        self.selected_file = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the file picker."""
        with Container(classes="file-picker-container"):
            yield Static(self.title, classes="file-picker-header")
            
            yield Input(
                value=self.initial_path,
                placeholder="Path to config directory...",
                id="path-input",
                classes="file-picker-path"
            )
            
            yield DirectoryTree(
                self.initial_path,
                id="directory-tree",
                classes="file-picker-tree"
            )
            
            with Horizontal(classes="file-picker-footer"):
                yield Button("Cancel", id="cancel", classes="file-picker-button")
                yield Button("Load", id="load", classes="file-picker-button", variant="primary")
    
    def on_mount(self) -> None:
        pass

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection."""
        if event.path.suffix.lower() in self.allowed_extensions:
            self.selected_file = str(event.path)
            self.query_one("#path-input", Input).value = str(event.path.parent)
    
    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Handle directory selection."""
        self.query_one("#path-input", Input).value = str(event.path)
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle path input changes."""
        if event.input.id == "path-input":
            path = Path(event.value)
            if path.exists() and path.is_dir():
                tree = self.query_one("#directory-tree", DirectoryTree)
                tree.path = str(path)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.action_cancel()
        elif event.button.id == "load":
            self.action_confirm()
    
    def action_cancel(self) -> None:
        """Cancel file selection."""
        self.dismiss(None)
    
    def action_confirm(self) -> None:
        """Confirm file selection."""
        if self.selected_file and Path(self.selected_file).exists():
            self.dismiss(self.selected_file)
        else:
            # Try to find a file in the current directory
            path_input = self.query_one("#path-input", Input)
            current_path = Path(path_input.value)
            if current_path.is_file() and current_path.suffix.lower() in self.allowed_extensions:
                self.dismiss(str(current_path))
            elif current_path.is_dir():
                # Look for files with allowed extensions in the directory
                matching_files = []
                for ext in self.allowed_extensions:
                    matching_files.extend(current_path.glob(f"*{ext}"))
                if matching_files:
                    # Use the first matching file found
                    self.dismiss(str(matching_files[0]))
                else:
                    ext_list = ", ".join(self.allowed_extensions)
                    self.notify(f"No files with extensions {ext_list} found in the selected directory", severity="error")
            else:
                ext_list = ", ".join(self.allowed_extensions)
                self.notify(f"Please select a file with one of these extensions: {ext_list}", severity="error")
