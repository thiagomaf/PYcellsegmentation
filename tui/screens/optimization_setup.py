from pathlib import Path
from typing import List
from textual.app import ComposeResult
from textual.screen import Screen, ModalScreen
from textual.widgets import Header, Footer, Button, Label, Input, Checkbox, Static, Switch
from textual.containers import Vertical, Horizontal, Grid, Container
from textual.message import Message

try:
    from tui.optimization.models import OptimizationProject, ParameterRanges
except Exception as e:
    raise
from tui.screens.file_picker import FilePicker

class SaveProjectModal(ModalScreen[str]):
    """Modal to enter a filename for the new optimization project."""
    
    def compose(self) -> ComposeResult:
        with Container(classes="modal-container"):
            yield Label("Enter filename for optimization project (.opt.json):")
            yield Input(placeholder="project_name.opt.json", id="filename-input")
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Save & Start", id="save", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "save":
            filename = self.query_one("#filename-input", Input).value
            if filename:
                if not filename.endswith(".opt.json"):
                    filename += ".opt.json"
                self.dismiss(filename)

class OptimizationSetup(Screen):
    """Screen for setting up a new optimization project."""
    
    CSS = """
    OptimizationSetup {
        align: center middle;
    }
    
    .setup-container {
        width: 90%;
        height: 90%;
        border: solid green;
        padding: 1 2;
        background: $surface;
    }
    
    .section-title {
        text-align: left;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
        width: 100%;
    }
    
    .config-list {
        height: 10;
        border: solid $secondary;
        overflow-y: scroll;
        margin-bottom: 1;
    }
    
    .param-row {
        height: auto;
        margin-bottom: 1;
        align: left middle;
    }
    
    .param-label {
        width: 20;
    }
    
    .param-input {
        width: 10;
        margin-right: 2;
    }

    .action-bar {
        dock: bottom;
        height: 3;
        margin-top: 2;
        align: center middle;
    }
    
    Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        try:
            yield Header()
            with Container(classes="setup-container"):
                yield Label("Project Setup", classes="title")
                yield Static("Configure parameter search space (optional - can be edited later)", classes="section-subtitle")
                
                # --- Parameter Configuration ---
                yield Label("Parameter Search Space", classes="section-title")
                
                with Vertical():
                    # Diameter
                    with Horizontal(classes="param-row"):
                        yield Checkbox("Diameter", value=True, id="opt-diameter")
                        yield Label("Range:", classes="param-label")
                        yield Input(value="15", id="diam-min", classes="param-input", type="integer")
                        yield Label("-")
                        yield Input(value="100", id="diam-max", classes="param-input", type="integer")

                    # Min Size
                    with Horizontal(classes="param-row"):
                        yield Checkbox("Min Size", value=True, id="opt-minsize")
                        yield Label("Range:", classes="param-label")
                        yield Input(value="5", id="size-min", classes="param-input", type="integer")
                        yield Label("-")
                        yield Input(value="50", id="size-max", classes="param-input", type="integer")

                    # Flow Threshold
                    with Horizontal(classes="param-row"):
                        yield Checkbox("Flow Thresh", value=False, id="opt-flow")
                        yield Label("Range:", classes="param-label")
                        yield Input(value="0.0", id="flow-min", classes="param-input", type="number")
                        yield Label("-")
                        yield Input(value="1.0", id="flow-max", classes="param-input", type="number")

                    # CellProb Threshold
                    with Horizontal(classes="param-row"):
                        yield Checkbox("CellProb Thresh", value=False, id="opt-cellprob")
                        yield Label("Range:", classes="param-label")
                        yield Input(value="-6.0", id="cp-min", classes="param-input", type="number")
                        yield Label("-")
                        yield Input(value="6.0", id="cp-max", classes="param-input", type="number")

                # --- Actions ---
                with Horizontal(classes="action-bar"):
                    yield Button("Load Existing Project", id="load-btn", variant="default")
                    yield Button("Create Project", id="start-btn", variant="success")
                    yield Button("Create with Defaults", id="create-defaults-btn", variant="default")
                    yield Button("Back", id="back-btn", variant="error")

            yield Footer()
        except Exception as e:
            raise

    def on_mount(self) -> None:
        """Initialize the setup screen."""
        # No initialization needed - projects can be created without configs
        # Configs and images can be added later in the project dashboard
        pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
            
        elif event.button.id == "load-btn":
            self.action_load_project()
            
        elif event.button.id == "start-btn":
            self.action_start_new()
        elif event.button.id == "create-defaults-btn":
            self.action_create_with_defaults()

    def action_load_project(self):
        """Open file picker to load .opt.json"""
        def on_selected(filepath):
            if filepath:
                try:
                    project = OptimizationProject.load(filepath)
                    self.launch_dashboard(project)
                except Exception as e:
                    self.notify(f"Error loading project: {e}", severity="error")
                    
        self.app.push_screen(FilePicker(), on_selected)

    def action_start_new(self):
        """Collect inputs and ask for filename."""
        # Ask for filename
        def on_filename(filename):
            if filename:
                self.create_and_start_project(filename, use_custom_ranges=True)
        
        self.app.push_screen(SaveProjectModal(), on_filename)
    
    def action_create_with_defaults(self):
        """Create project with default parameter ranges."""
        # Ask for filename
        def on_filename(filename):
            if filename:
                self.create_and_start_project(filename, use_custom_ranges=False)
        
        self.app.push_screen(SaveProjectModal(), on_filename)

    def create_and_start_project(self, filename: str, use_custom_ranges: bool = True):
        """Create the project object and save it."""
        try:
            if use_custom_ranges:
                # Parse parameters from UI
                ranges = ParameterRanges(
                    diameter_min=int(self.query_one("#diam-min", Input).value),
                    diameter_max=int(self.query_one("#diam-max", Input).value),
                    min_size_min=int(self.query_one("#size-min", Input).value),
                    min_size_max=int(self.query_one("#size-max", Input).value),
                    flow_threshold_min=float(self.query_one("#flow-min", Input).value),
                    flow_threshold_max=float(self.query_one("#flow-max", Input).value),
                    cellprob_threshold_min=float(self.query_one("#cp-min", Input).value),
                    cellprob_threshold_max=float(self.query_one("#cp-max", Input).value),
                    
                    optimize_diameter=self.query_one("#opt-diameter", Checkbox).value,
                    optimize_min_size=self.query_one("#opt-minsize", Checkbox).value,
                    optimize_flow_threshold=self.query_one("#opt-flow", Checkbox).value,
                    optimize_cellprob_threshold=self.query_one("#opt-cellprob", Checkbox).value,
                )
            else:
                # Use default parameter ranges
                ranges = ParameterRanges()
            
            project = OptimizationProject(
                filepath=filename,
                image_pool=[],  # Empty pool - can be populated later
                config_files=[],  # No configs initially
                parameter_ranges=ranges
            )
            
            project.save()
            self.notify(f"Project saved to {filename}")
            self.launch_dashboard(project)
            
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")
        except Exception as e:
            self.notify(f"Error creating project: {e}", severity="error")

    def launch_dashboard(self, project: OptimizationProject):
        """Switch to the project dashboard screen."""
        from tui.screens.project_dashboard import ProjectDashboard
        self.app.push_screen(ProjectDashboard(project))

