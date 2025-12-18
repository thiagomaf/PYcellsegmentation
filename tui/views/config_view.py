import os

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static

from tui.views.base_view import BaseView
from tui.models import ProjectConfig
from tui.widgets.general_view import GeneralView
from tui.widgets.images_view import ImagesView
from tui.widgets.parameters_view import ParametersView
from tui.widgets.preview_view import PreviewView

class ConfigView(BaseView):
    """CONFIG view with secondary sidebar for General/Images/Parameters/Preview."""

    # Class-level debug flag (can be overridden by instance)
    DEBUG = os.getenv("CONFIG_VIEW_DEBUG", "false").lower() == "true"
    BINDINGS = [
        Binding("1", "view_general",    "General",    tooltip="View the general settings"),
        Binding("2", "view_images",     "Images",     tooltip="View the images settings"),    
        Binding("3", "view_parameters", "Parameters", tooltip="View the parameters settings"),
        Binding("4", "view_preview",    "Preview",    tooltip="View the preview settings"),
        Binding("c", "pass", show=False),
        Binding("p", "pass", show=False),
        Binding("r", "pass", show=False),
    ]
    
    
    def __init__(self, config: ProjectConfig, filepath: str | None = None, debug: bool = False):
        super().__init__(config, filepath)
        self.current_subview = "general"
        # Use instance debug if provided, otherwise use class default
        self.debug = debug if debug is not None else self.DEBUG
        # Make container focusable so its bindings appear in footer
        self.can_focus = True
    
    def compose(self) -> ComposeResult:
        with Horizontal(id="config-container"):
            # Secondary sidebar
            with Vertical(id="config-secondary-sidebar"):
                yield Static("Navigation",   id="config-nav-title",      classes="nav-btn")

                yield Button("▶ General",    id="config-nav-general",    classes="nav-btn nav-btn--selected")
                yield Button("  Images",     id="config-nav-images",     classes="nav-btn")
                yield Button("  Parameters", id="config-nav-parameters", classes="nav-btn")
                yield Button("  Preview",    id="config-nav-preview",    classes="nav-btn")
            
            # Main content area (inherits from BaseView)
            with Horizontal(id="config-main-content"):
                # Upper section
                with Vertical(id="config-dashboard"):
                    yield Static("Config Summary", id="config-summary")
                    yield Static("Loading...", id="config-content")
    
    def on_mount(self) -> None:
        """Initialize with default subview."""
        # Focus this container so its bindings appear in the footer
        self.focus()
        # Use call_after_refresh to ensure widgets are fully mounted
        self.call_after_refresh(self.show_subview, self.current_subview)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle secondary sidebar button presses."""
        button_id = event.button.id

        if self.debug: self.notify(f"ConfigView button pressed: {button_id}", timeout=20)

        # Only handle buttons from the secondary sidebar
        if button_id and button_id.startswith("config-nav-"):
            event.stop()  # Stop event propagation to parent

            if self.debug: self.notify(f"ConfigView handling button: {button_id}", timeout=20)

            if button_id == "config-nav-general":
                self.show_subview("general")
            elif button_id == "config-nav-images":
                self.show_subview("images")
            elif button_id == "config-nav-parameters":
                self.show_subview("parameters")
            elif button_id == "config-nav-preview":
                self.show_subview("preview")
        else:
            print(f"ConfigView ignoring button: {button_id}")  # Debug
    
    def action_view_general(self) -> None:
        """Switch to general subview."""
        self.show_subview("general")
    
    def action_view_images(self) -> None:
        """Switch to images subview."""
        self.show_subview("images")
    
    def action_view_parameters(self) -> None:
        """Switch to parameters subview."""
        self.show_subview("parameters")
    
    def action_view_preview(self) -> None:
        """Switch to preview subview."""
        self.show_subview("preview")
    
    def show_subview(self, subview_name: str) -> None:
        """Switch between General/Images/Parameters/Preview."""
        if self.debug: self.notify(f"show_subview called with: {subview_name}", timeout=20)
        
        # Map subview names to widget classes
        widget_classes = {
            "general":    GeneralView,
            "images":     ImagesView,
            "parameters": ParametersView,
            "preview":    PreviewView
        }
        
        # Map subview names to button IDs
        button_ids = {
            "general":    "config-nav-general",
            "images":     "config-nav-images",
            "parameters": "config-nav-parameters",
            "preview":    "config-nav-preview"
        }
        
        # Map subview names to button labels (selected, unselected)
        button_labels = {
            "general":    ("▶ General",    "  General"),
            "images":     ("▶ Images",     "  Images"),
            "parameters": ("▶ Parameters", "  Parameters"),
            "preview":    ("▶ Preview",    "  Preview")
        }
        
        # Validate subview name
        if subview_name not in widget_classes:
            if self.debug: self.notify(f"Invalid subview name: {subview_name}", timeout=20)
            return
        
        # Update button selection
        for svname, bid in button_ids.items():
            try:
                button = self.query_one(f"#{bid}", Button)
                if svname == subview_name:
                    button.add_class("nav-btn--selected")
                    button.label = button_labels[svname][0]
                else:
                    button.remove_class("nav-btn--selected")
                    button.label = button_labels[svname][1]
            except Exception as e:
                if self.debug: self.notify(f"Button {bid} not found: {e}", timeout=20)
                continue
        
        # Remove current widget from lower section
        try:
            # Try to find the content area - it should be a Vertical container
            content_area = self.query_one("#config-dashboard", Vertical)
            if self.debug: self.notify(f"Found content area, removing children", timeout=20)
            content_area.remove_children()
        except Exception as e:
            # If content area not found, try alternative query
            try:
                # Try querying as Container instead
                content_area = self.query_one("#config-dashboard", Container)
                if self.debug: self.notify(f"Found content area (as Container), removing children", timeout=20)
                content_area.remove_children()
            except Exception as e2:
                # If still not found, show detailed error
                if self.debug: self.notify(f"Content area not found. Error 1: {e}, Error 2: {e2}", timeout=5)
                # Try to list available widgets for debugging
                try:
                    all_widgets = self.query("*")
                    widget_ids = [w.id for w in all_widgets if hasattr(w, 'id') and w.id]
                    if self.debug: self.notify(f"Available widget IDs: {widget_ids}", timeout=5)
                except:
                    pass
                return
        
        # Mount appropriate widget from tui.widgets
        try:
            widget_class = widget_classes[subview_name]
            if self.debug: self.notify(f"Instantiating {widget_class.__name__}", timeout=20)
            # All widget views take config as first parameter
            new_widget = widget_class(self.config)
            if self.debug: self.notify(f"Mounting widget to content area", timeout=20)
            content_area.mount(new_widget)
            self.current_subview = subview_name
            if self.debug: self.notify(f"Successfully switched to {subview_name}", timeout=20)
        except Exception as e:
            # If mounting fails, show error message
            error_msg = f"Error loading {subview_name} view: {e}"
            if self.debug: self.notify(error_msg, timeout=5)
            try:
                content_area.mount(Static(error_msg, id="error-message"))
            except Exception as e2:
                if self.debug: self.notify(f"Failed to show error: {e2}", timeout=20)
