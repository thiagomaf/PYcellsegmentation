"""Main TUI application entry point."""
import sys
from pathlib import Path

# Add the project root (parent of tui) to the path so imports work
# This must be done BEFORE importing from tui.* packages
if __name__ == "__main__":
    tui_dir = Path(__file__).parent
    project_root = tui_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Also patch if running as a module but path isn't set correctly (defensive)
# Actually, top level execution needs this. If we are imported, we assume path is correct?
# If we are imported as 'from tui.app import ...', tui must be in path.
# If we run as script, we need the patch.
# But 'if __name__ == "__main__":' only runs if executed directly.
# The imports below are at module level.
# If we run 'python tui/app.py', __name__ is "__main__".
# So the block above runs. THEN the imports below run.
# This should work.

from typing import Optional
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Footer, Header

# These imports depend on sys.path being correct
try:
    from tui.screens.dashboard import Dashboard
    from tui.models import ProjectConfig
except ModuleNotFoundError:
    # If the patch above didn't work (e.g. some weird execution mode), try patching unconditionally
    # This happens if we are running as script but somehow __name__ check passed but path insertion failed?
    # Or if we are NOT main but tui is not in path?
    # Let's just force the patch if we are in the tui dir
    tui_dir = Path(__file__).parent
    project_root = tui_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from tui.screens.dashboard import Dashboard
    from tui.models import ProjectConfig


class PyCellSegTUI(App):
    """Main TUI application for PyCellSegmentation."""
    
    CSS_PATH = str(Path(__file__).parent / "styles.tcss")
    
    TITLE = "PyCellSegmentation TUI"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("n", "new_project", "New Project"),
        ("l", "load_project", "Load Project"),
        ("f12",    "inspector", "Inspector"),  # Open widget inspector (F12 or Ctrl+I)
        ("ctrl+i", "inspector", "Inspector"),  # Alternative key binding
        ("escape", "back", "Back"),  # For project editor
        ("ctrl+s", "save", "Save"),  # For project editor
        ("1", "view_general", "General"),  # For project editor
        ("2", "view_images", "Images"),  # For project editor
        ("3", "view_parameters", "Parameters"),  # For project editor
        ("4", "view_preview", "Preview"),  # For project editor
    ]
    
    def __init__(self):
        """Initialize the application."""
        super().__init__()
        self.current_project: ProjectConfig | None = None
        self.current_filepath: str | None = None
        self._content_area: Container | None = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(id="app-header")
        yield Container(id="content-area")
        yield Footer(id="app-footer")
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Get reference to content area
        self._content_area = self.query_one("#content-area", Container)
        # Mount the dashboard in the content area
        self.show_dashboard()
    
    def show_dashboard(self) -> None:
        """Show the dashboard in the content area."""
        if self._content_area is None:
            self._content_area = self.query_one("#content-area", Container)
        
        # Clear content area and mount dashboard
        self._content_area.remove_children()
        dashboard = Dashboard()
        self._content_area.mount(dashboard)
        
        # Update header and footer to dashboard state
        self.update_header("Project Configuration Manager", "")
        self.update_footer([
            ("q", "Quit"),
            ("n", "New Project"),
            ("l", "Load Project"),
        ])
    
    def show_project_editor(self, config: Optional[ProjectConfig] = None, filepath: Optional[str] = None) -> None:
        """Show the project editor in the content area."""
        from tui.screens.project_editor import ProjectEditor
        
        if self._content_area is None:
            self._content_area = self.query_one("#content-area", Container)
        
        # Clear content area and mount project editor
        self._content_area.remove_children()
        editor = ProjectEditor(config, filepath)
        self._content_area.mount(editor)
        
        # Store current project info
        self.current_project = config or ProjectConfig()
        self.current_filepath = filepath
        
        # Update header and footer
        project_name = "New Project" if filepath is None else Path(filepath).name
        self.update_header("PyCellSegmentation TUI", f"Editing: {project_name}")
        self.update_footer([
            ("escape", "Back"),
            ("ctrl+s", "Save"),
            ("1-4", "Switch views"),
        ])
    
    def update_header(self, title: str | None, subtitle: str | None = None) -> None:
        """Update the app header title and subtitle.
        
        Args:
            title: Title to set, or None to leave unchanged
            subtitle: Subtitle to set. Use None to leave unchanged, or empty string "" to clear/remove the subtitle.
        """
        if title is not None:
            self.title = title
        
        # Handle subtitle: None = don't change, "" = clear, any string = set
        if subtitle is not None:
            self.sub_title = subtitle  # Empty string will clear it, any other string will set it
        # If subtitle is None, leave it unchanged
    
    def update_footer(self, bindings: list[tuple[str, str]]) -> None:
        """Update the footer with custom bindings.
        
        Args:
            bindings: List of (key, description) tuples
        """
        try:
            footer = self.query_one("#app-footer", Footer)
            # Footer in Textual shows app bindings, so we need to update app bindings
            # For now, we'll just update the footer's highlight_key to show current context
            # The actual bindings are managed by the app's BINDINGS
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "new-project":
            self.action_new_project()
        elif button_id == "load-project":
            self.action_load_project()
        elif button_id == "exit":
            self.action_quit()
    
    def action_new_project(self) -> None:
        """Create a new project."""
        self.show_project_editor()
    
    def action_load_project(self) -> None:
        """Load an existing project."""
        from tui.screens.file_picker import FilePicker
        
        def on_dismiss(filepath):
            if filepath:
                self.load_project(filepath)
        
        self.push_screen(FilePicker(), on_dismiss)
    
    def action_quit(self) -> None:
        """Exit the application."""
        self.exit()
    
    def action_back(self) -> None:
        """Go back to dashboard from project editor."""
        # Check if we're in project editor
        from tui.screens.project_editor import ProjectEditor
        try:
            project_editor = self.query_one(ProjectEditor)
            self.show_dashboard()
        except Exception:
            pass
    
    def action_save(self) -> None:
        """Save the current project (delegates to project editor)."""
        # Find project editor and call its save action
        from tui.screens.project_editor import ProjectEditor
        try:
            project_editor = self.query_one(ProjectEditor)
            project_editor.action_save()
        except Exception:
            pass
    
    def action_view_general(self) -> None:
        """Switch to general view (delegates to project editor)."""
        from tui.screens.project_editor import ProjectEditor
        try:
            project_editor = self.query_one(ProjectEditor)
            project_editor.action_view_general()
        except Exception:
            pass
    
    def action_view_images(self) -> None:
        """Switch to images view (delegates to project editor)."""
        from tui.screens.project_editor import ProjectEditor
        try:
            project_editor = self.query_one(ProjectEditor)
            project_editor.action_view_images()
        except Exception:
            pass
    
    def action_view_parameters(self) -> None:
        """Switch to parameters view (delegates to project editor)."""
        from tui.screens.project_editor import ProjectEditor
        try:
            project_editor = self.query_one(ProjectEditor)
            project_editor.action_view_parameters()
        except Exception:
            pass
    
    def action_view_preview(self) -> None:
        """Switch to preview view (delegates to project editor)."""
        from tui.screens.project_editor import ProjectEditor
        try:
            project_editor = self.query_one(ProjectEditor)
            project_editor.action_view_preview()
        except Exception:
            pass
    
    def action_inspector(self) -> None:
        """Open the widget inspector for debugging.
        
        When running with 'textual run --dev', this action is automatically
        provided by Textual. We fall back to a simple inspector if not available.
        """
        # First, try to use the parent's action_inspector if it exists
        # This will work when running with 'textual run --dev'
        parent_class = super()
        
        if hasattr(parent_class, 'action_inspector'):
            try:
                # Call the parent's method - this should work with textual run --dev
                return parent_class.action_inspector()
            except Exception:
                # If parent's method fails, try inspect() method
                pass
        
        # Try the inspect() method if it exists (alternative API)
        if hasattr(self, 'inspect'):
            try:
                self.inspect()
                return
            except Exception:
                pass
        
        # If we get here, the inspector is not available from textual-dev
        # Try our fallback simple inspector
        try:
            from tui.widgets.simple_inspector import SimpleInspector
            inspector = SimpleInspector()
            # Store reference to the current screen so inspector can access it
            inspector._target_screen = self.screen
            self.push_screen(inspector)
            return
        except Exception as e:
            # If fallback fails, show error message
            self.notify(
                "⚠️ Inspector Not Available\n\n"
                "The full inspector is ONLY available when you run:\n\n"
                "  textual run --dev tui/app.py\n\n"
                "For now, a simple inspector was attempted but failed.\n"
                "Please use 'textual run --dev' for full inspector features.",
                severity="warning",
                timeout=25
            )
    
    def load_project(self, filepath: str) -> None:
        """Load a project from a file."""
        try:
            config = ProjectConfig.from_json_file(filepath)
            self.show_project_editor(config, filepath)
        except Exception as e:
            self.notify(f"Error loading project: {e}", severity="error")


def main():
    """Main entry point for the TUI."""
    app = PyCellSegTUI()
    app.run()


if __name__ == "__main__":
    main()
