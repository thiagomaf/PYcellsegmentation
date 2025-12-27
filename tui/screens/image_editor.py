"""Image editor modal screen."""
# from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Select, Static, Switch

from tui.models import ImageConfiguration, SegmentationOptions, RescalingConfig, TilingParameters


class ImageEditor(ModalScreen[ImageConfiguration]):
    """Modal screen for editing image configuration."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+s", "save", "Save"),
    ]
    
    def __init__(self, image_config: ImageConfiguration):
        """Initialize the image editor."""
        super().__init__()
        self.image_config = image_config
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the image editor."""
        with Container(classes="image-editor-container"):
            yield Static("Edit Image Configuration", classes="editor-header")
            
            with Vertical(classes="editor-content"):
                # Basic Info Section
                with Container(classes="form-section"):
                    yield Static("Basic Information", classes="form-section-title")
                    
                    with Horizontal(classes="form-row"):
                        yield Static("Image ID:", classes="form-label")
                        yield Input(
                            self.image_config.image_id,
                            id="image-id",
                            classes="form-input"
                        )
                    
                    with Horizontal(classes="form-row"):
                        yield Static("Image Filename:", classes="form-label")
                        yield Input(
                            self.image_config.original_image_filename,
                            id="image-filename",
                            classes="form-input"
                        )
                    
                    with Horizontal(classes="form-row"):
                        yield Static("MPP X:", classes="form-label")
                        yield Input(
                            str(self.image_config.mpp_x) if self.image_config.mpp_x else "",
                            id="mpp-x",
                            classes="form-input",
                            placeholder="0.2125"
                        )
                    
                    with Horizontal(classes="form-row"):
                        yield Static("MPP Y:", classes="form-label")
                        yield Input(
                            str(self.image_config.mpp_y) if self.image_config.mpp_y else "",
                            id="mpp-y",
                            classes="form-input",
                            placeholder="0.2125"
                        )
                    
                    with Horizontal(classes="form-row"):
                        yield Static("Active:", classes="form-label")
                        yield Switch(
                            self.image_config.is_active,
                            id="is-active"
                        )
                
                # Segmentation Options
                with Collapsible(title="Segmentation Options", collapsed=False):
                    with Container(classes="form-section"):
                        with Horizontal(classes="form-row"):
                            yield Static("Apply Segmentation:", classes="form-label")
                            yield Switch(
                                self.image_config.segmentation_options.apply_segmentation,
                                id="apply-segmentation"
                            )
                        
                        # Rescaling Config
                        with Collapsible(title="Rescaling Configuration", collapsed=True):
                            rescaling = self.image_config.segmentation_options.rescaling_config
                            with Container(classes="form-section"):
                                with Horizontal(classes="form-row"):
                                    yield Static("Scale Factor:", classes="form-label")
                                    yield Input(
                                        str(rescaling.scale_factor) if rescaling else "1.0",
                                        id="scale-factor",
                                        classes="form-input",
                                        placeholder="1.0"
                                    )
                                
                                with Horizontal(classes="form-row"):
                                    yield Static("Interpolation:", classes="form-label")
                                    yield Select(
                                        [
                                            ("INTER_NEAREST", "INTER_NEAREST"),
                                            ("INTER_LINEAR", "INTER_LINEAR"),
                                            ("INTER_AREA", "INTER_AREA"),
                                            ("INTER_CUBIC", "INTER_CUBIC"),
                                            ("INTER_LANCZOS4", "INTER_LANCZOS4"),
                                        ],
                                        value=rescaling.interpolation if rescaling else "INTER_LINEAR",
                                        id="interpolation",
                                        classes="form-input"
                                    )
                        
                        # Tiling Parameters
                        with Collapsible(title="Tiling Parameters", collapsed=True):
                            tiling = self.image_config.segmentation_options.tiling_parameters
                            with Container(classes="form-section"):
                                with Horizontal(classes="form-row"):
                                    yield Static("Apply Tiling:", classes="form-label")
                                    yield Switch(
                                        tiling.apply_tiling if tiling else False,
                                        id="apply-tiling"
                                    )
                                
                                with Horizontal(classes="form-row"):
                                    yield Static("Tile Size:", classes="form-label")
                                    yield Input(
                                        str(tiling.tile_size) if tiling and tiling.tile_size else "",
                                        id="tile-size",
                                        classes="form-input",
                                        placeholder="2000"
                                    )
                                
                                with Horizontal(classes="form-row"):
                                    yield Static("Overlap:", classes="form-label")
                                    yield Input(
                                        str(tiling.overlap) if tiling and tiling.overlap else "100",
                                        id="tile-overlap",
                                        classes="form-input",
                                        placeholder="100"
                                    )
            
            with Horizontal(classes="editor-footer"):
                yield Static("", classes="toolbar-spacer")
                yield Button("Save", id="save", classes="toolbar-button", variant="primary")
                yield Button("Cancel", id="cancel", classes="toolbar-button")                
    
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
        """Save the image configuration."""
        try:
            # Get basic info
            image_id = self.query_one("#image-id", Input).value
            filename = self.query_one("#image-filename", Input).value
            is_active = self.query_one("#is-active", Switch).value
            
            # Get MPP values
            mpp_x_str = self.query_one("#mpp-x", Input).value
            mpp_y_str = self.query_one("#mpp-y", Input).value
            mpp_x = float(mpp_x_str) if mpp_x_str else None
            mpp_y = float(mpp_y_str) if mpp_y_str else None
            
            # Get segmentation options
            apply_segmentation = self.query_one("#apply-segmentation", Switch).value
            
            # Get rescaling config
            scale_factor_str = self.query_one("#scale-factor", Input).value
            scale_factor = float(scale_factor_str) if scale_factor_str else 1.0
            interpolation = self.query_one("#interpolation", Select).value
            
            rescaling_config = RescalingConfig(
                scale_factor=scale_factor,
                interpolation=interpolation
            ) if scale_factor != 1.0 else None
            
            # Get tiling parameters
            apply_tiling = self.query_one("#apply-tiling", Switch).value
            tile_size_str = self.query_one("#tile-size", Input).value
            tile_size = int(tile_size_str) if tile_size_str and apply_tiling else None
            overlap_str = self.query_one("#tile-overlap", Input).value
            overlap = int(overlap_str) if overlap_str else 100
            
            tiling_params = TilingParameters(
                apply_tiling=apply_tiling,
                tile_size=tile_size,
                overlap=overlap
            ) if apply_tiling else None
            
            segmentation_options = SegmentationOptions(
                apply_segmentation=apply_segmentation,
                rescaling_config=rescaling_config,
                tiling_parameters=tiling_params
            )
            
            # Create updated image configuration
            updated_config = ImageConfiguration(
                image_id=image_id,
                original_image_filename=filename,
                is_active=is_active,
                mpp_x=mpp_x,
                mpp_y=mpp_y,
                segmentation_options=segmentation_options
            )
            
            self.dismiss(updated_config)
        except Exception as e:
            self.notify(f"Error saving: {e}", severity="error")
