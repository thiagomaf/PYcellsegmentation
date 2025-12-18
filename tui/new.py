import sys
from pathlib import Path
from typing import Optional
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Header, Button, Footer

# Add the project root (parent of tui) to the path so imports work
# This must be done BEFORE importing from tui.* packages
if __name__ == "__main__":
    tui_dir = Path(__file__).parent
    project_root = tui_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

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
    
    TITLE = "PyCellSegmentation"
    BINDINGS = [
        Binding("l",      "load_project", "Load Project", tooltip="Load a project from a file"),
        Binding("q",      "quit",         "Quit",         tooltip="Quit the application"),
        Binding("escape", "back",         "Back",         tooltip="Go back to dashboard"),
        Binding("f12",    "inspector",    "Inspector",    tooltip="Open the widget inspector"),
        Binding("ctrl+i", "inspector",    "Inspector",    tooltip="Open the widget inspector"),
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
    
    # SHOW CONTENT SCREENS -------------------------------------------------------------
    def show_dashboard(self) -> None:
        """Show the dashboard in the content area."""
        if self._content_area is None:
            self._content_area = self.query_one("#content-area", Container)
        
        # Clear content area and mount dashboard
        self._content_area.remove_children()
        dashboard = Dashboard()
        self._content_area.mount(dashboard)
        
        # # Update header and footer to dashboard state
        # self.update_header("Project Configuration Manager", "")

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
        
        # # Update header and footer
        # project_name = "New Project" if filepath is None else Path(filepath).name
        # self.update_header("PyCellSegmentation TUI", f"Editing: {project_name}")
    

    # DEFINE CONTROLS ------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "new-project":
            self.action_new_project()
        elif button_id == "load-project":
            self.action_load_project()
        elif button_id == "exit":
            self.action_quit()

    # DEFINE ACTIONS -------------------------------------------------------------------
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
    
    def action_back(self) -> None:
        """Go back to dashboard from project editor."""
        from tui.screens.project_editor import ProjectEditor
        try:
            # Check if we're in project editor
            project_editor = self.query_one(ProjectEditor)
            self.show_dashboard()
        except Exception:
            # If not in project editor, do nothing
            pass
    
    def action_quit(self) -> None:
        """Exit the application."""
        self.exit()

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