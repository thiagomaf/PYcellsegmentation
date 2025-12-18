"""Parameter editor modal screen."""
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static, Switch

from tui.models import ParameterConfiguration, CellposeParameters


class ParameterEditor(ModalScreen[ParameterConfiguration]):
    """Modal screen for editing parameter configuration."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+s", "save", "Save"),
    ]
    
    def __init__(self, param_config: ParameterConfiguration):
        """Initialize the parameter editor."""
        super().__init__()
        self.param_config = param_config
        self.cp_params = param_config.cellpose_parameters
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the parameter editor."""
        with Container(classes="param-editor-container"):
            yield Static("Edit Parameter Configuration", classes="editor-header")
            
            with Vertical():
                # Basic Info
                with Container(classes="form-section"):
                    yield Static("Basic Information", classes="form-section-title")
                    
                    with Container(classes="form-row"):
                        yield Static("Parameter Set ID:", classes="form-label")
                        yield Input(
                            self.param_config.param_set_id,
                            id="param-set-id",
                            classes="form-input"
                        )
                    
                    with Container(classes="form-row"):
                        yield Static("Active:", classes="form-label")
                        yield Switch(
                            self.param_config.is_active,
                            id="is-active"
                        )
                
                # Cellpose Parameters
                with Container(classes="form-section"):
                    yield Static("Cellpose Parameters", classes="form-section-title")
                    
                    with Container(classes="form-row"):
                        yield Static("Model Choice:", classes="form-label")
                        yield Select(
                            [
                                ("cyto3", "cyto3"),
                                ("nuclei", "nuclei"),
                                ("cyto2", "cyto2"),
                                ("cyto", "cyto"),
                            ],
                            value=self.cp_params.MODEL_CHOICE,
                            id="model-choice",
                            classes="form-input"
                        )
                    
                    with Container(classes="form-row"):
                        yield Static("Diameter:", classes="form-label")
                        yield Input(
                            str(self.cp_params.DIAMETER) if self.cp_params.DIAMETER else "",
                            id="diameter",
                            classes="form-input",
                            placeholder="60"
                        )
                    
                    with Container(classes="form-row"):
                        yield Static("Min Size:", classes="form-label")
                        yield Input(
                            str(self.cp_params.MIN_SIZE),
                            id="min-size",
                            classes="form-input",
                            placeholder="15"
                        )
                    
                    with Container(classes="form-row"):
                        yield Static("Cellprob Threshold:", classes="form-label")
                        yield Input(
                            str(self.cp_params.CELLPROB_THRESHOLD),
                            id="cellprob-threshold",
                            classes="form-input",
                            placeholder="0.0"
                        )
                    
                    with Container(classes="form-row"):
                        yield Static("Force Grayscale:", classes="form-label")
                        yield Switch(
                            self.cp_params.FORCE_GRAYSCALE,
                            id="force-grayscale"
                        )
                    
                    with Container(classes="form-row"):
                        yield Static("Z Projection Method:", classes="form-label")
                        yield Select(
                            [
                                ("max", "max"),
                                ("mean", "mean"),
                                ("none", "none"),
                            ],
                            value=self.cp_params.Z_PROJECTION_METHOD or "max",
                            id="z-projection",
                            classes="form-input",
                            allow_blank=True
                        )
                    
                    with Container(classes="form-row"):
                        yield Static("Channel Index:", classes="form-label")
                        yield Input(
                            str(self.cp_params.CHANNEL_INDEX),
                            id="channel-index",
                            classes="form-input",
                            placeholder="0"
                        )
                    
                    with Container(classes="form-row"):
                        yield Static("Enable 3D Segmentation:", classes="form-label")
                        yield Switch(
                            self.cp_params.ENABLE_3D_SEGMENTATION,
                            id="enable-3d"
                        )
            
            with Horizontal(classes="editor-footer"):
                yield Button("Save", id="save", classes="footer-button", variant="primary")
                yield Button("Cancel", id="cancel", classes="footer-button")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.action_cancel()
        elif event.button.id == "save":
            self.action_save()
    
    def action_cancel(self) -> None:
        """Cancel editing."""
        self.dismiss(None)
    
    def action_save(self) -> None:
        """Save the parameter configuration."""
        try:
            # Get basic info
            param_set_id = self.query_one("#param-set-id", Input).value
            is_active = self.query_one("#is-active", Switch).value
            
            # Get Cellpose parameters
            model_choice = self.query_one("#model-choice", Select).value
            
            diameter_str = self.query_one("#diameter", Input).value
            diameter = int(diameter_str) if diameter_str else None
            
            min_size_str = self.query_one("#min-size", Input).value
            min_size = int(min_size_str) if min_size_str else 15
            
            cellprob_str = self.query_one("#cellprob-threshold", Input).value
            cellprob = float(cellprob_str) if cellprob_str else 0.0
            
            force_grayscale = self.query_one("#force-grayscale", Switch).value
            
            z_projection = self.query_one("#z-projection", Select).value
            if z_projection == "":
                z_projection = None
            
            channel_index_str = self.query_one("#channel-index", Input).value
            channel_index = int(channel_index_str) if channel_index_str else 0
            
            enable_3d = self.query_one("#enable-3d", Switch).value
            
            # Create updated parameter configuration
            cp_params = CellposeParameters(
                MODEL_CHOICE=model_choice,
                DIAMETER=diameter,
                MIN_SIZE=min_size,
                CELLPROB_THRESHOLD=cellprob,
                FORCE_GRAYSCALE=force_grayscale,
                Z_PROJECTION_METHOD=z_projection,
                CHANNEL_INDEX=channel_index,
                ENABLE_3D_SEGMENTATION=enable_3d
            )
            
            updated_config = ParameterConfiguration(
                param_set_id=param_set_id,
                is_active=is_active,
                cellpose_parameters=cp_params
            )
            
            self.dismiss(updated_config)
        except Exception as e:
            self.notify(f"Error saving: {e}", severity="error")
