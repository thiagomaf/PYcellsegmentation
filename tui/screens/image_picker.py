"""Image picker screen for adding images."""
from pathlib import Path
from typing import List
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Static, Label
from textual import events

class ImagePicker(ModalScreen[List[str]]):
    """Modal screen for picking image files."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "confirm", "Add Selected"),
    ]
    
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ndpi", ".svs"}
    
    def __init__(self, initial_path: str = None):
        """Initialize the image picker."""
        super().__init__()
        if initial_path is None:
            # Default to project root or user home
            project_root = Path(__file__).parent.parent.parent
            self.initial_path = str(project_root)
        else:
            self.initial_path = initial_path
        self.selected_path: Path | None = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the file picker."""
        with Container(classes="file-picker-container"):
            yield Static("Select Image(s) or Directory", classes="file-picker-header")
            
            yield Input(
                value=self.initial_path,
                placeholder="Path to image directory...",
                id="path-input",
                classes="file-picker-path"
            )
            
            yield DirectoryTree(
                self.initial_path,
                id="directory-tree",
                classes="file-picker-tree"
            )
            
            with Container(classes="file-picker-info"):
                yield Label("Select a file to add it, or a directory to add all images within it.", classes="text-muted")
                yield Label("", id="selection-status")
            
            with Horizontal(classes="file-picker-footer"):
                yield Button("Cancel", id="cancel", classes="file-picker-button")
                yield Button("Add Selected", id="add", classes="file-picker-button", variant="primary")
    
    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection."""
        self.selected_path = event.path
        self.query_one("#path-input", Input).value = str(event.path.parent)
        self._update_status()
        
    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Handle directory selection."""
        self.selected_path = event.path
        self.query_one("#path-input", Input).value = str(event.path)
        self._update_status()
    
    def _update_status(self) -> None:
        """Update the status label based on selection."""
        status_label = self.query_one("#selection-status", Label)
        if not self.selected_path:
            status_label.update("No selection")
            return
            
        if self.selected_path.is_file():
            if self.selected_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                status_label.update(f"Selected: {self.selected_path.name}")
            else:
                status_label.update(f"Warning: {self.selected_path.name} may not be a supported image")
        elif self.selected_path.is_dir():
            # Count images
            try:
                images = [p for p in self.selected_path.iterdir() 
                         if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS]
                status_label.update(f"Directory: {self.selected_path.name} ({len(images)} images found)")
            except Exception:
                status_label.update(f"Directory: {self.selected_path.name}")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle path input changes."""
        if event.input.id == "path-input":
            path = Path(event.value)
            if path.exists():
                if path.is_dir():
                    tree = self.query_one("#directory-tree", DirectoryTree)
                    try:
                        tree.path = str(path)
                    except Exception:
                        pass
                self.selected_path = path
                self._update_status()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.action_cancel()
        elif event.button.id == "add":
            self.action_confirm()
    
    def action_cancel(self) -> None:
        """Cancel selection."""
        self.dismiss([])
    
    def action_confirm(self) -> None:
        """Confirm selection."""
        if not self.selected_path or not self.selected_path.exists():
            self.notify("Please select a valid file or directory", severity="error")
            return
            
        results = []
        if self.selected_path.is_file():
            results.append(str(self.selected_path))
        elif self.selected_path.is_dir():
            # Add all images in directory
            try:
                for p in self.selected_path.iterdir():
                    if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS:
                        results.append(str(p))
            except Exception as e:
                self.notify(f"Error reading directory: {e}", severity="error")
                return
        
        if not results:
            self.notify("No valid images selected", severity="warning")
            return
            
        self.dismiss(results)

