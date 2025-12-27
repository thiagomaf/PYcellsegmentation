"""Screen for editing parameter ranges in a project."""
from pathlib import Path
import logging
from textual.app import ComposeResult
from textual.screen import Screen, ModalScreen
from textual.widgets import Header, Footer, Button, Label, Input, Checkbox, Static
from textual.containers import Vertical, Horizontal, Container, ScrollableContainer
from textual.reactive import reactive

from tui.optimization.models import ParameterRanges, OptimizationProject, SuggestedParameterPoint
# Lazy import visualization to avoid importing numpy at module level
# from tui.optimization.visualization import setup_parameter_coverage_plot, generate_grid_samples
from tui.models import ProjectConfig, ParameterConfiguration, CellposeParameters, ImageConfiguration, SegmentationOptions

logger = logging.getLogger(__name__)

try:
    from textual_plotext import PlotextPlot
    HAS_TEXTUAL_PLOTEXT = True
except ImportError:
    HAS_TEXTUAL_PLOTEXT = False

try:
    from tui.widgets.plotext_static import PlotextStatic
    HAS_PLOTEXT_STATIC = True
except ImportError:
    HAS_PLOTEXT_STATIC = False

HAS_PLOTEXT = HAS_TEXTUAL_PLOTEXT or HAS_PLOTEXT_STATIC

class ParameterRangesEditor(ModalScreen[bool]):
    """Screen for editing parameter ranges."""
    
    CSS = """
    ParameterRangesEditor {
        align: center middle;
    }
    
    .editor-container {
        width: 90%;
        height: 90%;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
        layout: horizontal;
    }
    
    .left-panel {
        width: 1fr;
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    .params-container {
        height: auto;
        margin-bottom: 1;
    }
    
    .right-panel {
        width: 1fr;
        height: 1fr;
        padding: 1;
        border-left: solid $primary;
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
    
    .plot-container {
        height: 1fr;
        border: solid $secondary;
        padding: 1;
    }
    
    #coverage-plot {
        height: 1fr;
        width: 100%;
        min-height: 20;
        border: solid $secondary;
    }
    """
    
    def __init__(self, ranges: ParameterRanges, project: OptimizationProject = None):
        super().__init__()
        self.ranges = ranges
        self.project = project
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the editor."""
        yield Header()
        with Container(classes="editor-container"):
            with ScrollableContainer(classes="left-panel"):
                yield Label("Edit Parameter Ranges", classes="title")
                
                with Vertical(classes="params-container"):
                    # Diameter
                    with Horizontal(classes="param-row"):
                        yield Checkbox("Diameter", value=self.ranges.optimize_diameter, id="opt-diameter")
                        # yield Label("Range:", classes="param-label")
                        yield Input(value=str(self.ranges.diameter_min), id="diam-min", classes="param-input", type="integer")
                        yield Label("-")
                        yield Input(value=str(self.ranges.diameter_max), id="diam-max", classes="param-input", type="integer")

                    # Min Size
                    with Horizontal(classes="param-row"):
                        yield Checkbox("Min Size", value=self.ranges.optimize_min_size, id="opt-minsize")
                        # yield Label("Range:", classes="param-label")
                        yield Input(value=str(self.ranges.min_size_min), id="size-min", classes="param-input", type="integer")
                        yield Label("-")
                        yield Input(value=str(self.ranges.min_size_max), id="size-max", classes="param-input", type="integer")

                    # Flow Threshold - REMOVED: Not available in PYcellsegmentation pipeline
                    # with Horizontal(classes="param-row"):
                    #     yield Checkbox("Flow Thresh", value=self.ranges.optimize_flow_threshold, id="opt-flow")
                    #     yield Input(value=str(self.ranges.flow_threshold_min), id="flow-min", classes="param-input", type="number")
                    #     yield Label("-")
                    #     yield Input(value=str(self.ranges.flow_threshold_max), id="flow-max", classes="param-input", type="number")

                    # CellProb Threshold
                    with Horizontal(classes="param-row"):
                        yield Checkbox("CellProb Thresh", value=self.ranges.optimize_cellprob_threshold, id="opt-cellprob")
                        # yield Label("Range:", classes="param-label")
                        yield Input(value=str(self.ranges.cellprob_threshold_min), id="cp-min", classes="param-input", type="number")
                        yield Label("-")
                        yield Input(value=str(self.ranges.cellprob_threshold_max), id="cp-max", classes="param-input", type="number")

                # Search space population (only if project is available)
                if self.project:
                    yield Label("Search Space Population", classes="title")
                    with Horizontal():
                        yield Input(value="10", id="num-samples", classes="param-input", placeholder="Number of points")
                        yield Button("Populate Search Space", id="populate-btn", variant="primary")
                    
                    with Horizontal():
                        yield Button("Generate Config", id="generate-config-btn", variant="success")
                        yield Static(f"Suggested points: {len(self.project.suggested_points)}", id="suggested-count")
                
                # Actions
                with Horizontal():
                    yield Button("Save", id="save-btn", variant="success")
                    yield Button("Cancel", id="cancel-btn", variant="default")
            
            # Visualization panel
            with Container(classes="right-panel"):
                yield Label("Parameter Space Coverage", classes="title")
                with Container(classes="plot-container", id="plot-container"):
                    if HAS_PLOTEXT_STATIC:
                        # Use direct plotext for better color control
                        yield PlotextStatic(id="coverage-plot")
                    elif HAS_TEXTUAL_PLOTEXT:
                        yield PlotextPlot(id="coverage-plot")
                    else:
                        yield Static("plotext not installed. Install with: pip install plotext", id="plot-placeholder")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PARAM1","location":"parameter_ranges_editor.py:127","message":"ParameterRangesEditor.on_mount() entry","data":{"has_project":self.project is not None},"timestamp":__import__("time").time()*1000})+"\n")
        except: pass
        # #endregion
        try:
            # Clear suggested points when opening/refreshing the view
            if self.project:
                self.project.suggested_points.clear()
                self.project.save()  # Save the cleared state
                # Update suggested count display
                count_widget = self.query_one("#suggested-count", Static)
                count_widget.update(f"Suggested points: {len(self.project.suggested_points)}")
            
            # FIX: Use call_after_refresh to ensure widgets are ready and layout is calculated
            # This ensures the plot widget has a valid size before we try to render
            self.call_after_refresh(self.update_visualization)
        except Exception as e:
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json, traceback
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PARAM1","location":"parameter_ranges_editor.py:133","message":"ParameterRangesEditor.on_mount() exception","data":{"error":str(e),"traceback":traceback.format_exc()},"timestamp":__import__("time").time()*1000})+"\n")
            except: pass
            # #endregion
            self.notify(f"Error initializing visualization: {e}", severity="error")
    
    def _show_empty_plot(self, plt, title: str, xlabel: str, ylabel: str = "") -> None:
        """Helper: Show an empty plot with a message."""
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.grid(True)
        plt.scatter([0], [0], marker='+', color='blue', label='')
    
    def _calculate_axis_limits_with_padding(self, x_min: float, x_max: float, y_min: float, y_max: float, padding: float = 0.1) -> tuple:
        """Helper: Calculate axis limits with padding."""
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if x_range > 0:
            x_padding = x_range * padding
            x_lim_min = x_min - x_padding
            x_lim_max = x_max + x_padding
        else:
            x_lim_min = x_min - 0.1
            x_lim_max = x_max + 0.1
        
        if y_range > 0:
            y_padding = y_range * padding
            y_lim_min = y_min - y_padding
            y_lim_max = y_max + y_padding
        else:
            y_lim_min = y_min - 0.1
            y_lim_max = y_max + 0.1
        
        return (x_lim_min, x_lim_max, y_lim_min, y_lim_max)
    
    def _normalize_parameter_column(self, col, param_min: float, param_max: float):
        """Helper: Normalize a parameter column to [0, 1].
        
        Args:
            col: numpy array of parameter values
            param_min: minimum value for normalization
            param_max: maximum value for normalization
            
        Returns:
            Normalized numpy array
        """
        import numpy as np
        if param_max == param_min or abs(param_max - param_min) < 1e-10:
            return np.full_like(col, 0.5, dtype=float)
        else:
            param_range = param_max - param_min
            return (col - param_min) / param_range
    
    def _display_plot(self, plot_widget, plt) -> None:
        """Helper: Build and display the plot in the widget."""
        plot_text = plt.build()
        if plot_text:
            from rich.text import Text
            try:
                rich_text = Text.from_ansi(plot_text)
                plot_widget.update(rich_text)
            except:
                plot_widget.update(plot_text)
            plot_widget.refresh()
    
    def _filter_unique_points(
        self, 
        candidate_samples: list, 
        existing_points: list, 
        num_samples: int,
        parameter_ranges: ParameterRanges
    ) -> list:
        """Filter candidate samples to ensure uniqueness and good space coverage.
        
        Args:
            candidate_samples: List of candidate parameter dictionaries
            existing_points: List of existing parameter dictionaries to avoid
            num_samples: Number of unique samples to return
            parameter_ranges: Parameter ranges for normalization
            
        Returns:
            List of unique parameter dictionaries that maximize coverage
        """
        import numpy as np
        
        if not candidate_samples:
            return []
        
        # Determine active parameters for normalization
        active_params = []
        if parameter_ranges.optimize_diameter:
            active_params.append(('diameter', parameter_ranges.diameter_min, parameter_ranges.diameter_max))
        if parameter_ranges.optimize_min_size:
            active_params.append(('min_size', parameter_ranges.min_size_min, parameter_ranges.min_size_max))
        if parameter_ranges.optimize_cellprob_threshold:
            active_params.append(('cellprob_threshold', parameter_ranges.cellprob_threshold_min, parameter_ranges.cellprob_threshold_max))
        
        if not active_params:
            return candidate_samples[:num_samples]
        
        # Normalize all points to [0, 1] for distance calculation
        def normalize_point(params: dict) -> np.ndarray:
            """Convert parameter dict to normalized numpy array."""
            point = []
            for param_name, param_min, param_max in active_params:
                val = params.get(param_name, param_min)
                # Normalize to [0, 1]
                if param_max == param_min:
                    norm_val = 0.0
                else:
                    norm_val = (val - param_min) / (param_max - param_min)
                point.append(norm_val)
            return np.array(point)
        
        # Normalize existing points
        existing_normalized = [normalize_point(p) for p in existing_points] if existing_points else []
        
        # Normalize candidate points
        candidate_normalized = [normalize_point(p) for p in candidate_samples]
        
        # Filter candidates that are too close to existing points
        # Use a threshold based on the parameter space size
        # For LHS, we want points that are well-separated
        threshold = 0.05  # 5% of normalized space - ensures good separation
        
        filtered_candidates = []
        filtered_candidate_normalized = []
        
        for cand, cand_norm in zip(candidate_samples, candidate_normalized):
            # Check minimum distance to any existing point
            if existing_normalized:
                min_dist = min(np.linalg.norm(cand_norm - exist_norm) for exist_norm in existing_normalized)
                if min_dist < threshold:
                    continue  # Too close to existing point, skip
            filtered_candidates.append(cand)
            filtered_candidate_normalized.append(cand_norm)
        
        # If we don't have enough filtered candidates, relax the threshold
        if len(filtered_candidates) < num_samples:
            # Try with a smaller threshold
            threshold = 0.02  # 2% of normalized space
            filtered_candidates = []
            filtered_candidate_normalized = []
            for cand, cand_norm in zip(candidate_samples, candidate_normalized):
                if existing_normalized:
                    min_dist = min(np.linalg.norm(cand_norm - exist_norm) for exist_norm in existing_normalized)
                    if min_dist < threshold:
                        continue
                filtered_candidates.append(cand)
                filtered_candidate_normalized.append(cand_norm)
        
        # If still not enough, use all candidates (they're already LHS, so well-distributed)
        if len(filtered_candidates) < num_samples:
            filtered_candidates = candidate_samples
            filtered_candidate_normalized = candidate_normalized
        
        # Now select points that maximize minimum distance from each other and existing points
        # This ensures good space coverage
        selected = []
        selected_normalized = []
        remaining = filtered_candidates.copy()
        remaining_normalized = filtered_candidate_normalized.copy()
        
        # Start with the point farthest from all existing points
        if remaining:
            if existing_normalized:
                # Find candidate with maximum minimum distance to existing points
                best_idx = max(
                    range(len(remaining)),
                    key=lambda i: min(
                        np.linalg.norm(remaining_normalized[i] - exist_norm) 
                        for exist_norm in existing_normalized
                    )
                )
            else:
                # No existing points, just pick first
                best_idx = 0
            
            selected.append(remaining.pop(best_idx))
            selected_normalized.append(remaining_normalized.pop(best_idx))
        
        # Greedily select remaining points that maximize minimum distance
        while len(selected) < num_samples and remaining:
            best_idx = None
            best_min_dist = -1
            
            for i, cand_norm in enumerate(remaining_normalized):
                # Calculate minimum distance to all selected and existing points
                all_points = selected_normalized + existing_normalized
                min_dist = min(np.linalg.norm(cand_norm - pt) for pt in all_points)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i
            
            if best_idx is not None:
                selected.append(remaining.pop(best_idx))
                selected_normalized.append(remaining_normalized.pop(best_idx))
            else:
                break
        
        return selected[:num_samples]
    
    def update_visualization(self) -> None:
        """Update the parameter coverage visualization."""
        if not HAS_PLOTEXT:
            logger.warning("Plotext not available - cannot show visualization")
            return
        
        try:
            # Try to use PlotextStatic first (direct plotext)
            if HAS_PLOTEXT_STATIC:
                try:
                    plot_widget = self.query_one("#coverage-plot", PlotextStatic)
                    plt = plot_widget.plt
                    
                    # STEP 8: Better error handling - clear everything including colors
                    try:
                        plt.clear_data()
                        plt.clear_figure()
                        plt.clear_color()  # Clear color state too
                    except Exception as e:
                        logger.warning(f"Error clearing plot: {e}")
                        # Continue anyway - try to clear individually
                        try:
                            plt.clear_data()
                        except:
                            pass
                        try:
                            plt.clear_figure()
                        except:
                            pass
                    
                    # Check for numpy (needed for PCA and some operations)
                    try:
                        import numpy as np
                    except ImportError:
                        self._show_empty_plot(plt, "numpy is required for visualization.\nInstall with: pip install numpy", "")
                        # Use refresh_plot instead of _display_plot
                        def update_plot():
                            try:
                                plot_widget.refresh_plot()
                            except Exception as e:
                                logger.error(f"Error refreshing plot: {e}", exc_info=True)
                        self.call_after_refresh(update_plot)
                        return
                    
                    # Set plot size
                    if hasattr(plot_widget, 'size') and plot_widget.size.width > 0 and plot_widget.size.height > 0:
                        plt.plotsize(int(plot_widget.size.width), int(plot_widget.size.height))
                    else:
                        plt.plotsize(70, 30)
                    
                    # Determine which parameters are active
                    active_params = []
                    if self.ranges.optimize_diameter:
                        active_params.append(('diameter', self.ranges.diameter_min, self.ranges.diameter_max))
                    if self.ranges.optimize_min_size:
                        active_params.append(('min_size', self.ranges.min_size_min, self.ranges.min_size_max))
                    if self.ranges.optimize_cellprob_threshold:
                        active_params.append(('cellprob_threshold', self.ranges.cellprob_threshold_min, self.ranges.cellprob_threshold_max))
                    
                    # Extract real data from project, grouped by config file
                    x_label = "X"
                    y_label = "Y"
                    config_data = {}  # {config_name: {'x': [...], 'y': [...]}}
                    
                    if len(active_params) >= 2:
                        # STEP 9: Create color/marker mapping upfront for consistency
                        from pathlib import Path
                        color_names = ['blue', 'red', 'green', 'yellow', 'magenta', 'cyan', 'white', 'orange',
                                     'blue+', 'red+', 'green+', 'yellow+', 'magenta+', 'cyan+', 'white+', 'orange+']
                        markers = ['+', 'x', '*', 'o', 'dot', 'braille', 'hd', 'sd']
                        config_color_map = {}
                        config_marker_map = {}
                        
                        # Collect all parameter data first
                        all_data_points = []  # List of lists: each inner list is [p1, p2, p3, ...] for all active params
                        labels = []  # Config name for each point
                        
                        if self.project and self.project.config_files:
                            from tui.optimization.visualization import extract_parameters_from_config
                            
                            # First pass: collect all config names and assign colors/markers consistently
                            config_names_seen = []
                            for config_info in self.project.config_files:
                                if config_info.included:
                                    config_name = Path(config_info.filepath).stem
                                    if config_name not in config_names_seen:
                                        config_names_seen.append(config_name)
                                        idx = len(config_names_seen) - 1
                                        config_color_map[config_name] = color_names[idx % len(color_names)]
                                        config_marker_map[config_name] = markers[idx % len(markers)]
                            
                            # Second pass: collect data
                            for config_info in self.project.config_files:
                                if config_info.included:
                                    config_name = Path(config_info.filepath).stem
                                    params_list = extract_parameters_from_config(config_info.filepath)
                                    for params in params_list:
                                        # Collect all active parameter values in order
                                        point = []
                                        all_present = True
                                        for param_name, _, _ in active_params:
                                            if param_name in params:
                                                point.append(params[param_name])
                                            else:
                                                all_present = False
                                                break
                                        if all_present and len(point) == len(active_params):
                                            all_data_points.append(point)
                                            labels.append(config_name)
                                            # Also store in config_data for simple 2D case
                                            if config_name not in config_data:
                                                config_data[config_name] = {'x': [], 'y': []}
                                            config_data[config_name]['x'].append(point[0])
                                            config_data[config_name]['y'].append(point[1] if len(point) > 1 else 0)
                        
                        # Determine visualization approach
                        if len(active_params) == 2:
                            # Simple 2D: use first two parameters directly
                            param1_name, param1_min, param1_max = active_params[0]
                            param2_name, param2_min, param2_max = active_params[1]
                            x_label = param1_name.replace('_', ' ').title()
                            y_label = param2_name.replace('_', ' ').title()
                            
                            # Rebuild config_data from all_data_points for 2D case
                            config_data = {}
                            for i, point in enumerate(all_data_points):
                                config_name = labels[i]
                                if config_name not in config_data:
                                    config_data[config_name] = {'x': [], 'y': []}
                                config_data[config_name]['x'].append(point[0])
                                config_data[config_name]['y'].append(point[1])
                        else:
                            # More than 2 parameters: use PCA for dimension reduction
                            try:
                                from sklearn.decomposition import PCA
                                import numpy as np
                                
                                if not all_data_points:
                                    raise ValueError("No data points for PCA")
                                
                                # Convert to numpy array
                                data_matrix = np.array(all_data_points)
                                
                                # Normalize parameters to [0, 1] for PCA (since they have different scales)
                                normalized_data = []
                                for i, (param_name, param_min, param_max) in enumerate(active_params):
                                    col = data_matrix[:, i]
                                    normalized_col = self._normalize_parameter_column(col, param_min, param_max)
                                    normalized_data.append(normalized_col)
                                normalized_matrix = np.column_stack(normalized_data)
                                
                                # Apply PCA
                                pca = PCA(n_components=2)
                                pca_result = pca.fit_transform(normalized_matrix)
                                
                                # Convert PCA results to lists
                                all_x_pca = pca_result[:, 0].tolist()
                                all_y_pca = pca_result[:, 1].tolist()
                                
                                # Rebuild config_data with PCA coordinates
                                config_data = {}
                                for i, (x_val, y_val) in enumerate(zip(all_x_pca, all_y_pca)):
                                    config_name = labels[i]
                                    if config_name not in config_data:
                                        config_data[config_name] = {'x': [], 'y': []}
                                    config_data[config_name]['x'].append(x_val)
                                    config_data[config_name]['y'].append(y_val)
                                
                                # Get axis ranges from PCA results with padding
                                x_min_pca, x_max_pca = float(np.min(pca_result[:, 0])), float(np.max(pca_result[:, 0]))
                                y_min_pca, y_max_pca = float(np.min(pca_result[:, 1])), float(np.max(pca_result[:, 1]))
                                
                                x_lim_min, x_lim_max, y_lim_min, y_lim_max = self._calculate_axis_limits_with_padding(
                                    x_min_pca, x_max_pca, y_min_pca, y_max_pca
                                )
                                plt.xlim(x_lim_min, x_lim_max)
                                plt.ylim(y_lim_min, y_lim_max)
                                
                                # Create axis labels showing which parameters contribute most
                                pc1_loadings = np.abs(pca.components_[0])
                                pc2_loadings = np.abs(pca.components_[1])
                                pc1_idx = np.argmax(pc1_loadings)
                                pc2_idx = np.argmax(pc2_loadings)
                                
                                pc1_param = active_params[pc1_idx][0].replace('_', ' ').title()
                                pc2_param = active_params[pc2_idx][0].replace('_', ' ').title()
                                
                                var1 = pca.explained_variance_ratio_[0] * 100
                                var2 = pca.explained_variance_ratio_[1] * 100
                                
                                x_label = f"PC1 ({pc1_param}, {var1:.1f}% var)"
                                y_label = f"PC2 ({pc2_param}, {var2:.1f}% var)"
                                
                                param_names = [p[0].replace('_', ' ').title() for p in active_params]
                                title = f"Parameter Space (PCA): {', '.join(param_names)}"
                                plt.title(title)
                                
                            except ImportError:
                                self._show_empty_plot(plt, "Need sklearn for multi-parameter visualization.\nInstall with: pip install scikit-learn", "")
                                # Use refresh_plot instead of _display_plot
                                def update_plot():
                                    try:
                                        plot_widget.refresh_plot()
                                    except Exception as e:
                                        logger.error(f"Error refreshing plot: {e}", exc_info=True)
                                self.call_after_refresh(update_plot)
                                return
                            except Exception as e:
                                logger.error(f"PCA failed: {e}", exc_info=True)
                                self._show_empty_plot(plt, f"Error in PCA: {str(e)[:100]}", "")
                                # Use refresh_plot instead of _display_plot
                                def update_plot():
                                    try:
                                        plot_widget.refresh_plot()
                                    except Exception as e:
                                        logger.error(f"Error refreshing plot: {e}", exc_info=True)
                                self.call_after_refresh(update_plot)
                                return
                        
                        # Set axis limits from ranges with padding (only for 2D case)
                        if len(active_params) == 2:
                            param1_name, param1_min, param1_max = active_params[0]
                            param2_name, param2_min, param2_max = active_params[1]
                            
                            x_lim_min, x_lim_max, y_lim_min, y_lim_max = self._calculate_axis_limits_with_padding(
                                param1_min, param1_max, param2_min, param2_max
                            )
                            plt.xlim(x_lim_min, x_lim_max)
                            plt.ylim(y_lim_min, y_lim_max)
                    
                    # Plot data with different colors for each config file
                    # STEP 9: Use consistent color/marker mapping created earlier
                    # STEP 8: Handle case where we have active params but no data
                    if not config_data and len(active_params) >= 2:
                        # No data points found in config files
                        self._show_empty_plot(plt, "No parameter data found in config files", 
                                            "Config files may not contain\nactive parameter sets")
                    elif config_data:
                        for config_name, data in config_data.items():
                            if data['x'] and data['y']:
                                # Use the consistent mapping created earlier
                                color = config_color_map.get(config_name, 'blue')
                                marker = config_marker_map.get(config_name, '+')
                                plt.scatter(data['x'], data['y'], marker=marker, color=color, label=config_name)
                        
                        # Add suggested points if available
                        if self.project and self.project.suggested_points:
                            suggested_x = []
                            suggested_y = []
                            suggested_data_points = []
                            
                            for suggested in self.project.suggested_points:
                                params = suggested.parameters
                                # Collect all active parameter values
                                point = []
                                all_present = True
                                for param_name, _, _ in active_params:
                                    if param_name in params:
                                        point.append(params[param_name])
                                    else:
                                        all_present = False
                                        break
                                
                                if all_present and len(point) == len(active_params):
                                    suggested_data_points.append(point)
                                    
                                    if len(active_params) == 2:
                                        # Simple 2D case
                                        suggested_x.append(point[0])
                                        suggested_y.append(point[1])
                            
                            # If PCA case, transform suggested points
                            if len(active_params) > 2 and suggested_data_points:
                                try:
                                    import numpy as np
                                    suggested_matrix = np.array(suggested_data_points)
                                    
                                    # Normalize suggested points same way as config points
                                    normalized_suggested = []
                                    for i, (param_name, param_min, param_max) in enumerate(active_params):
                                        col = suggested_matrix[:, i]
                                        normalized_col = self._normalize_parameter_column(col, param_min, param_max)
                                        normalized_suggested.append(normalized_col)
                                    normalized_suggested_matrix = np.column_stack(normalized_suggested)
                                    
                                    # Refit PCA on combined data
                                    if all_data_points:
                                        from sklearn.decomposition import PCA
                                        all_normalized = np.vstack([normalized_matrix, normalized_suggested_matrix])
                                        pca_combined = PCA(n_components=2)
                                        pca_combined_result = pca_combined.fit_transform(all_normalized)
                                        suggested_pca = pca_combined_result[len(normalized_matrix):]
                                        suggested_x = suggested_pca[:, 0].tolist()
                                        suggested_y = suggested_pca[:, 1].tolist()
                                        
                                        # Update axis ranges to include suggested points
                                        all_x_combined_pca = all_x_pca + suggested_x
                                        all_y_combined_pca = all_y_pca + suggested_y
                                        
                                        x_min_new = min(all_x_combined_pca)
                                        x_max_new = max(all_x_combined_pca)
                                        y_min_new = min(all_y_combined_pca)
                                        y_max_new = max(all_y_combined_pca)
                                        
                                        x_lim_min, x_lim_max, y_lim_min, y_lim_max = self._calculate_axis_limits_with_padding(
                                            x_min_new, x_max_new, y_min_new, y_max_new
                                        )
                                        plt.xlim(x_lim_min, x_lim_max)
                                        plt.ylim(y_lim_min, y_lim_max)
                                    else:
                                        # No config data, but we have suggested points - use PCA on suggested points only
                                        from sklearn.decomposition import PCA
                                        pca_suggested = PCA(n_components=2)
                                        pca_suggested_result = pca_suggested.fit_transform(normalized_suggested_matrix)
                                        suggested_x = pca_suggested_result[:, 0].tolist()
                                        suggested_y = pca_suggested_result[:, 1].tolist()
                                        
                                        # Set axis limits based on suggested points only
                                        x_min_new = min(suggested_x)
                                        x_max_new = max(suggested_x)
                                        y_min_new = min(suggested_y)
                                        y_max_new = max(suggested_y)
                                        
                                        x_lim_min, x_lim_max, y_lim_min, y_lim_max = self._calculate_axis_limits_with_padding(
                                            x_min_new, x_max_new, y_min_new, y_max_new
                                        )
                                        plt.xlim(x_lim_min, x_lim_max)
                                        plt.ylim(y_lim_min, y_lim_max)
                                except Exception as e:
                                    logger.error(f"Error transforming suggested points for PCA: {e}", exc_info=True)
                            
                            # Plot suggested points (for both 2D and PCA cases)
                            if suggested_x and suggested_y:
                                # For 2D case, update axis ranges to include suggested points
                                if len(active_params) == 2:
                                    # Collect all x and y from all config files
                                    all_x_combined = []
                                    all_y_combined = []
                                    for data in config_data.values():
                                        all_x_combined.extend(data['x'])
                                        all_y_combined.extend(data['y'])
                                    all_x_combined.extend(suggested_x)
                                    all_y_combined.extend(suggested_y)
                                    
                                    # Get current limits and add padding
                                    x_min_current = min(all_x_combined)
                                    x_max_current = max(all_x_combined)
                                    y_min_current = min(all_y_combined)
                                    y_max_current = max(all_y_combined)
                                    
                                    x_lim_min, x_lim_max, y_lim_min, y_lim_max = self._calculate_axis_limits_with_padding(
                                        x_min_current, x_max_current, y_min_current, y_max_current
                                    )
                                    plt.xlim(x_lim_min, x_lim_max)
                                    plt.ylim(y_lim_min, y_lim_max)
                                
                                # Plot suggested points with distinct marker/color
                                try:
                                    plt.scatter(suggested_x, suggested_y, marker='dot', color=3, label='Suggested')  # 3 = yellow
                                except:
                                    # Fallback to RGB yellow
                                    plt.scatter(suggested_x, suggested_y, marker='dot', color=(255, 255, 0), label='Suggested')
                        
                        # Better title format matching old code
                        if 'title' not in locals():
                            # Only set if not already set (PCA case sets it)
                            title = f"Parameter Space: {x_label} vs {y_label}"
                            plt.title(title)
                        plt.xlabel(x_label)
                        plt.ylabel(y_label)
                        plt.grid(True)
                    elif len(active_params) < 2:
                        # STEP 8: Better empty state with visible plot
                        self._show_empty_plot(plt, "Need at least 2 active parameters\nto visualize coverage",
                                             "Enable at least 2 parameters\nin the checkboxes above")
                    else:
                        # STEP 8: Better empty state with visible plot (no config files)
                        self._show_empty_plot(plt, "No config files to visualize",
                                             "Add config files to the project\nin the project dashboard")
                    
                    # Build and display using refresh_plot (has proper mounting checks)
                    # Use call_after_refresh to ensure widget is fully ready
                    def update_plot():
                        try:
                            plot_widget.refresh_plot()
                        except Exception as e:
                            logger.error(f"Error refreshing plot: {e}", exc_info=True)
                    
                    self.call_after_refresh(update_plot)
                    
                    return
                except Exception as e:
                    logger.error(f"PlotextStatic failed: {e}", exc_info=True)
                    try:
                        plot_widget = self.query_one("#coverage-plot", PlotextStatic)
                        plot_widget.update(f"Error: {str(e)[:100]}")
                    except:
                        pass
                    return
            
            # Fall back to PlotextPlot (textual-plotext wrapper)
            if HAS_TEXTUAL_PLOTEXT:
                plot_widget = self.query_one("#coverage-plot", PlotextPlot)
                plt = plot_widget.plt
                
                # Set up the plot using our visualization function
                # Lazy import to avoid loading numpy at module level
                from tui.optimization.visualization import setup_parameter_coverage_plot
                setup_parameter_coverage_plot(
                    plt,
                    self.ranges,
                    project=self.project
                )
                
                # Trigger a refresh of the plot widget
                plot_widget.refresh()
                
        except Exception as e:
            import traceback
            error_msg = f"Error generating visualization: {e}\n{traceback.format_exc()}"
            self.notify(error_msg, severity="error")
            logger.error(f"Error generating visualization: {e}")
            logger.debug(traceback.format_exc())
            # Try to show error in plot widget
            try:
                if HAS_PLOTEXT_STATIC:
                    plot_widget = self.query_one("#coverage-plot", PlotextStatic)
                    plot_widget.plt.title(f"Error: {str(e)[:100]}")
                    plot_widget.refresh_plot()
                elif HAS_TEXTUAL_PLOTEXT:
                    plot_widget = self.query_one("#coverage-plot", PlotextPlot)
                    plot_widget.plt.title(f"Error: {str(e)[:100]}")
                    plot_widget.refresh()
            except:
                pass
    
    def on_input_changed(self, event) -> None:
        """Update visualization when inputs change."""
        # Update ranges from UI first
        try:
            self._update_ranges_from_ui()
        except Exception as e:
            logger.warning(f"Error updating ranges from UI: {e}")
        # Debounce: only update after user stops typing
        self.set_timer(0.5, self.update_visualization)
    
    def on_checkbox_changed(self, event) -> None:
        """Update visualization when checkboxes change."""
        # Update ranges from UI first to reflect checkbox changes
        try:
            self._update_ranges_from_ui()
        except Exception as e:
            logger.warning(f"Error updating ranges from UI: {e}")
        # Then update visualization
        self.update_visualization()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.dismiss(False)
        elif event.button.id == "save-btn":
            self.action_save()
        elif event.button.id == "populate-btn":
            self.action_populate_search_space()
        elif event.button.id == "generate-config-btn":
            self.action_generate_config()
    
    def action_populate_search_space(self) -> None:
        """Generate LHS samples and add them as suggested points."""
        if not self.project:
            self.notify("No project loaded", severity="error")
            return
        
        try:
            # Get number of samples from input
            num_samples_str = self.query_one("#num-samples", Input).value
            num_samples = int(num_samples_str) if num_samples_str else 10
            
            if num_samples < 1:
                self.notify("Number of samples must be at least 1", severity="error")
                return
            
            # Update ranges from UI first
            self._update_ranges_from_ui()
            
            # Collect existing parameter points from config files AND existing suggested points
            existing_points = []
            if self.project:
                from tui.optimization.visualization import extract_parameters_from_config
                # Get points from config files
                for config_info in self.project.config_files:
                    if config_info.included:
                        params_list = extract_parameters_from_config(config_info.filepath)
                        existing_points.extend(params_list)
                # Also include existing suggested points
                for suggested in self.project.suggested_points:
                    existing_points.append(suggested.parameters)
            
            # Use Latin Hypercube Sampling (LHS) for better space-filling coverage
            # Generate more candidates than needed, then filter to ensure uniqueness
            from tui.optimization.visualization import generate_lhs_samples
            import numpy as np
            
            # Generate 3x more candidates to ensure we can filter out duplicates/overlaps
            candidate_multiplier = 3
            num_candidates = max(num_samples * candidate_multiplier, 50)  # At least 50 candidates
            
            # Generate LHS candidates
            candidate_samples = generate_lhs_samples(self.ranges, num_candidates)
            
            if not candidate_samples:
                self.notify("No active parameters to sample. Enable at least one parameter for optimization.", severity="warning")
                return
            
            # Filter candidates to ensure they're unique and cover new areas
            samples = self._filter_unique_points(
                candidate_samples, 
                existing_points, 
                num_samples,
                self.ranges
            )
            
            if not samples:
                self.notify("Could not generate unique points. Try increasing parameter ranges or reducing number of samples.", severity="warning")
                return
            
            if not samples:
                self.notify("No active parameters to sample. Enable at least one parameter for optimization.", severity="warning")
                return
            
            # Add as suggested points
            self.project.suggested_points = [SuggestedParameterPoint(parameters=s) for s in samples]
            
            # Update visualization
            self.update_visualization()
            
            # Update count display
            count_widget = self.query_one("#suggested-count", Static)
            count_widget.update(f"Suggested points: {len(self.project.suggested_points)}")
            
            self.notify(f"Generated {len(samples)} suggested parameter points", severity="success")
            
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")
        except Exception as e:
            self.notify(f"Error generating samples: {e}", severity="error")
    
    def action_generate_config(self) -> None:
        """Generate a config file from all suggested points."""
        if not self.project:
            self.notify("No project loaded", severity="error")
            return
        
        if not self.project.suggested_points:
            self.notify("No suggested points available. Populate search space first.", severity="warning")
            return
        
        if not self.project.image_pool:
            self.notify("No images in pool. Add images to the project first.", severity="warning")
            return
        
        try:
            # Create a new config file
            config = ProjectConfig()
            
            # Add images from the project pool
            for img_entry in self.project.image_pool:
                if img_entry.is_active:
                    # Create image configuration
                    img_config = ImageConfiguration(
                        image_id=Path(img_entry.filepath).stem,
                        original_image_filename=img_entry.filepath,
                        is_active=True,
                        segmentation_options=SegmentationOptions()
                    )
                    config.image_configurations.append(img_config)
            
            # Create parameter configurations from ALL suggested points
            for idx, suggested in enumerate(self.project.suggested_points):
                params = suggested.parameters
                
                # Create parameter configuration from suggested point
                cp_params = CellposeParameters(
                    MODEL_CHOICE="cyto3",
                    DIAMETER=int(params.get('diameter', 30)),
                    MIN_SIZE=int(params.get('min_size', 15)),
                    CELLPROB_THRESHOLD=float(params.get('cellprob_threshold', 0.0)),
                    FORCE_GRAYSCALE=True,
                    Z_PROJECTION_METHOD="max",
                    CHANNEL_INDEX=0,
                    ENABLE_3D_SEGMENTATION=False
                )
                
                param_config = ParameterConfiguration(
                    param_set_id=f"grid_{len(self.project.config_files) + 1}_{idx + 1}",
                    is_active=True,
                    cellpose_parameters=cp_params
                )
                config.cellpose_parameter_configurations.append(param_config)
            
            # Determine config file path - use /config folder in project root
            try:
                from src.file_paths import PROJECT_ROOT
                config_dir = Path(PROJECT_ROOT) / "config"
            except (ImportError, Exception):
                # Fallback: use project file location or current directory
                if self.project.filepath:
                    project_root = Path(self.project.filepath).parent.parent.parent
                else:
                    project_root = Path(__file__).parent.parent.parent
                config_dir = project_root / "config"
            
            # Ensure config directory exists
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique config filename
            config_filename = f"processing_config_grid_{len(self.project.config_files) + 1}.json"
            config_path = config_dir / config_filename
            
            # Save config file
            config.to_json_file(str(config_path))
            
            # Add to project config files (created_at will be set automatically by ConfigFileInfo)
            from tui.optimization.models import ConfigFileInfo
            config_info = ConfigFileInfo(filepath=str(config_path), included=True)
            self.project.config_files.append(config_info)
            
            # Clear all suggested points (they're now in the config)
            num_points = len(self.project.suggested_points)
            self.project.suggested_points.clear()
            
            # Save the project to persist the new config file
            self.project.save()
            
            # Update visualization
            self.update_visualization()
            
            # Update count display
            count_widget = self.query_one("#suggested-count", Static)
            count_widget.update(f"Suggested points: {len(self.project.suggested_points)}")
            
            self.notify(
                f"Generated config file with {num_points} parameter sets: {config_filename}",
                severity="success"
            )
            
        except Exception as e:
            self.notify(f"Error generating config: {e}", severity="error")
    
    def _update_ranges_from_ui(self) -> None:
        """Update ranges from UI inputs."""
        try:
            self.ranges.diameter_min = int(self.query_one("#diam-min", Input).value)
            self.ranges.diameter_max = int(self.query_one("#diam-max", Input).value)
            self.ranges.min_size_min = int(self.query_one("#size-min", Input).value)
            self.ranges.min_size_max = int(self.query_one("#size-max", Input).value)
            # Flow Threshold removed - not available in pipeline
            self.ranges.cellprob_threshold_min = float(self.query_one("#cp-min", Input).value)
            self.ranges.cellprob_threshold_max = float(self.query_one("#cp-max", Input).value)
            # Update checkbox states - this is critical for visualization refresh
            self.ranges.optimize_diameter = self.query_one("#opt-diameter", Checkbox).value
            self.ranges.optimize_min_size = self.query_one("#opt-minsize", Checkbox).value
            self.ranges.optimize_flow_threshold = False  # Always disabled
            self.ranges.optimize_cellprob_threshold = self.query_one("#opt-cellprob", Checkbox).value
        except Exception as e:
            logger.warning(f"Error updating ranges from UI: {e}")
            # Don't raise - allow partial updates if some fields fail
    
    def action_save(self) -> None:
        """Save the parameter ranges."""
        try:
            # Update ranges from UI
            self._update_ranges_from_ui()
            
            # Update project's parameter ranges if project exists
            if self.project:
                self.project.parameter_ranges = self.ranges
            
            self.dismiss(True)  # Return True to indicate save was successful
            
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")
