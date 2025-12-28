
from typing import Optional
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Static

from tui.models import ProjectConfig

from tui.views.config_view import ConfigView
from tui.views.pipeline_view import PipelineView
from tui.views.results_view import ResultsView

class ProjectEditor(Container):
    """Main project editor container."""
    
    BINDINGS = [
        Binding("c",      "view_config",   "Config Editor",   tooltip="View the config editor"),
        Binding("p",      "view_pipeline", "Pipeline Status", tooltip="View the pipeline status"),
        Binding("r",      "view_results",  "Result Explorer", tooltip="View the result explorer"),
        Binding("ctrl+s", "save",          "Save",            tooltip="Save the current project"),
    ]

    def __init__(self, config: Optional[ProjectConfig] = None, filepath: Optional[str] = None):
        super().__init__()
        self.config = config or ProjectConfig()
        self.filepath = filepath
        self.current_view = "config"
        self._current_view_widget = None
        # Make container focusable so its bindings appear in footer
        self.can_focus = True
    
    def compose(self) -> ComposeResult:
        with Horizontal(id="editor-container"):
            # Primary sidebar
            with Vertical(id="editor-sidebar"):
                yield Static("Navigation",        id="nav-title",    classes="nav-btn")

                yield Button("▶ Config Editor",   id="nav-config",   classes="nav-btn nav-btn--selected")
                yield Button("  Pipeline Status", id="nav-pipeline", classes="nav-btn", disabled=False)
                yield Button("  Result Explorer", id="nav-results",  classes="nav-btn", disabled=False)
            
            # Main content area (will be populated by show_view)
            with Container(id="editor-main-content"):
                yield Static("Loading...", id="content-placeholder")
    
    def on_mount(self) -> None:
        """Initialize with selected view."""
        # Focus this container so its bindings appear in the footer
        self.focus()
        # self.show_view("config")
        pass
    
    def show_view(self, view_name: str) -> None:
        """Switch between CONFIG, PIPELINE, RESULTS views."""
        # Map view names to view classes
        view_classes = {
            "config":   ConfigView,
            "pipeline": PipelineView,
            "results":  ResultsView
        }
        
        # Map view names to button IDs
        button_ids = {
            "config":   "nav-config",
            "pipeline": "nav-pipeline",
            "results":  "nav-results"
        }
        
        # Map view names to button labels (selected, unselected)
        button_labels = {
            "config":   ("▶ Config Editor",   "  Config Editor"),
            "pipeline": ("▶ Pipeline Status", "  Pipeline Status"),
            "results":  ("▶ Result Explorer", "  Result Explorer")
        }
        
        # Validate view name
        if view_name not in view_classes:
            return
        
        # Update primary sidebar button selection
        for vname, bid in button_ids.items():
            try:
                button = self.query_one(f"#{bid}", Button)
                if vname == view_name:
                    button.add_class("nav-btn--selected")
                    button.label = button_labels[vname][0]
                else:
                    button.remove_class("nav-btn--selected")
                    button.label = button_labels[vname][1]
            except Exception:
                continue
        
        # Remove current view widget
        try:
            content_area = self.query_one("#editor-main-content", Container)
            content_area.remove_children()
        except Exception:
            # If content area not found, cannot proceed
            return
        
        # Mount new view widget
        try:
            view_class = view_classes[view_name]
            new_view = view_class(self.config, self.filepath)
            content_area.mount(new_view)
            self._current_view_widget = new_view
            # Re-focus ProjectEditor after mounting view so its bindings (like escape) remain active
            # The view will still be able to receive focus for its own bindings when needed
            self.call_after_refresh(self.focus)
        except Exception as e:
            # If mounting fails, show error message
            try:
                content_area.mount(Static(f"Error loading view: {e}", id="error-message"))
            except Exception:
                pass
            return
        
        # Update self.current_view
        self.current_view = view_name
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle primary sidebar button presses."""
        button_id = event.button.id
        
        # print(f"ProjectEditor button pressed: {button_id}")  # Debug
        self.notify(f"ProjectEditor button pressed: {button_id}", timeout=2)
        
        # Only handle primary sidebar buttons, ignore others (let them bubble to child views)
        if button_id in ("nav-config", "nav-pipeline", "nav-results"):
            event.stop()  # Stop propagation after handling
            if button_id == "nav-config":
                self.show_view("config")
            elif button_id == "nav-pipeline":
                self.show_view("pipeline")
            elif button_id == "nav-results":
                self.show_view("results")
        # If it's not one of our buttons, don't handle it - let it bubble to child views

    def action_view_config(self) -> None:
        """Switch to config view."""
        self.show_view("config")
    
    def action_view_pipeline(self) -> None:
        """Switch to pipeline view."""
        self.show_view("pipeline")
    
    def action_view_results(self) -> None:
        """Switch to results view."""
        self.show_view("results")

    
    def action_save(self) -> None:
        """Save the current project."""
        from tui.screens.save_dialog import SaveDialog
        
        def on_dismiss(filepath):
            if filepath:
                # Update the filepath if it was changed
                self.filepath = filepath
                # Notify that save was successful
                self.app.notify(f"Project saved to {filepath}", severity="success")
        
        # Show save dialog with current config and filepath
        self.app.push_screen(SaveDialog(self.config, self.filepath), on_dismiss)


class ProjectEditorScreen(Screen):
    """Screen wrapper for ProjectEditor container."""
    
    BINDINGS = [
        ("escape", "back", "Back"),
    ]
    
    def __init__(self, config: Optional[ProjectConfig] = None, filepath: Optional[str] = None):
        """Initialize the project editor screen."""
        super().__init__()
        self.config = config
        self.filepath = filepath
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield ProjectEditor(self.config, self.filepath)
    
    def action_back(self) -> None:
        """Go back to the previous screen."""
        self.app.pop_screen()