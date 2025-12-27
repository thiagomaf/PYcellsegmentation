"""Pydantic models for PyCellSegmentation configuration files."""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class GlobalSettings(BaseModel):
    """Global segmentation settings."""
    default_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_processes: int = Field(default=1, ge=1)
    FORCE_GRAYSCALE: bool = True
    USE_GPU_IF_AVAILABLE: bool = True


class RescalingConfig(BaseModel):
    """Image rescaling configuration."""
    scale_factor: float = Field(default=1.0, gt=0.0, le=1.0)
    interpolation: Literal[
        "INTER_NEAREST",
        "INTER_LINEAR",
        "INTER_AREA",
        "INTER_CUBIC",
        "INTER_LANCZOS4"
    ] = "INTER_LINEAR"


class TilingParameters(BaseModel):
    """Image tiling parameters."""
    apply_tiling: bool = False
    tile_size: Optional[int] = Field(default=None, gt=0)
    overlap: Optional[int] = Field(default=100, ge=0)
    tile_output_prefix_base: Optional[str] = None


class SegmentationOptions(BaseModel):
    """Segmentation options for an image."""
    apply_segmentation: bool = True
    rescaling_config: Optional[RescalingConfig] = None
    tiling_parameters: TilingParameters = Field(default_factory=TilingParameters)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SegmentationOptions":
        """Create from dictionary, handling missing optional fields."""
        rescaling = None
        if "rescaling_config" in data and data["rescaling_config"]:
            rescaling = RescalingConfig(**data["rescaling_config"])
        
        # tiling_parameters now has a default_factory, so we only override if data is provided
        tiling = None
        if "tiling_parameters" in data and data["tiling_parameters"]:
            tiling = TilingParameters(**data["tiling_parameters"])
        
        # If tiling is None, Pydantic will use the default_factory
        kwargs = {
            "apply_segmentation": data.get("apply_segmentation", True),
            "rescaling_config": rescaling,
        }
        if tiling is not None:
            kwargs["tiling_parameters"] = tiling
        
        return cls(**kwargs)


class ImageConfiguration(BaseModel):
    """Configuration for a single image."""
    image_id: str
    original_image_filename: str
    is_active: bool = True
    mpp_x: Optional[float] = Field(default=None, gt=0.0)
    mpp_y: Optional[float] = Field(default=None, gt=0.0)
    segmentation_options: SegmentationOptions = Field(default_factory=SegmentationOptions)


class CellposeParameters(BaseModel):
    """Cellpose segmentation parameters."""
    MODEL_CHOICE: Literal["cyto3", "nuclei", "cyto2", "cyto"] = "cyto3"
    DIAMETER: Optional[int] = Field(default=None, gt=0)
    MIN_SIZE: int = Field(default=15, gt=0)
    CELLPROB_THRESHOLD: float = Field(default=0.0, ge=-6.0, le=6.0)
    FORCE_GRAYSCALE: bool = True
    Z_PROJECTION_METHOD: Optional[Literal["max", "mean", "none"]] = "max"
    CHANNEL_INDEX: int = Field(default=0, ge=0)
    ENABLE_3D_SEGMENTATION: bool = False
    USE_GPU: Optional[bool] = True


class ParameterConfiguration(BaseModel):
    """A set of Cellpose parameters."""
    param_set_id: str
    is_active: bool = True
    cellpose_parameters: CellposeParameters = Field(default_factory=CellposeParameters)


class MappingTask(BaseModel):
    """Mapping task configuration."""
    image_id: str
    transcript_file: str


class ProjectConfig(BaseModel):
    """Root configuration model for a project."""
    global_segmentation_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    image_configurations: List[ImageConfiguration] = Field(default_factory=list)
    cellpose_parameter_configurations: List[ParameterConfiguration] = Field(default_factory=list)
    mapping_tasks: List[MappingTask] = Field(default_factory=list)

    @classmethod
    def from_json_file(cls, filepath: str) -> "ProjectConfig":
        """Load a project configuration from a JSON file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Pydantic should handle most of this, but we need to ensure
        # nested optional objects are handled correctly
        try:
            return cls(**data)
        except Exception as e:
            # Fallback: manually construct if automatic parsing fails
            # This handles cases where optional nested objects might be None
            image_configs = []
            for img_data in data.get("image_configurations", []):
                seg_opts_data = img_data.get("segmentation_options", {})
                if seg_opts_data:
                    seg_opts = SegmentationOptions.from_dict(seg_opts_data)
                    img_data["segmentation_options"] = seg_opts
                image_configs.append(ImageConfiguration(**img_data))
            data["image_configurations"] = image_configs
            
            param_configs = []
            for param_data in data.get("cellpose_parameter_configurations", []):
                param_configs.append(ParameterConfiguration(**param_data))
            data["cellpose_parameter_configurations"] = param_configs
            
            return cls(**data)

    def to_json_file(self, filepath: str) -> None:
        """Save the project configuration to a JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(mode='json', exclude_none=True), f, indent=2)

    def to_json(self) -> str:
        """Convert the project configuration to a JSON string."""
        import json
        return json.dumps(self.model_dump(mode='json', exclude_none=True), indent=2)
