import os
import logging
from pathlib import Path
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Button, Label, DataTable, Sparkline, Static, Select
from textual.containers import Vertical, Horizontal, Grid, Container, ScrollableContainer
from textual import work
from textual.worker import Worker, WorkerState

from tui.optimization.models import OptimizationProject, ParameterSetEntry, OptimizationResult
from tui.widgets.plotext_static import PlotextStatic
# Lazy import - FileBasedOptimizer will be imported in __init__ to avoid loading numpy/tifffile at module level
# from tui.optimization.optimizer import FileBasedOptimizer

logger = logging.getLogger(__name__)

class OptimizationDashboard(Screen):
    """Dashboard for monitoring and controlling the optimization process."""
    
    CSS_PATH = str(Path(__file__).parent / "optimization_dashboard.tcss")
    
    def on_scroll(self, event) -> None:
        """Handle scroll events to pause table updates during scrolling."""
        self._is_scrolling = True
        # Reset scroll flag after 500ms of no scrolling
        if self._scroll_timer:
            self._scroll_timer.stop()
        self._scroll_timer = self.set_timer(0.5, self._reset_scroll_flag)
    
    def _reset_scroll_flag(self) -> None:
        """Reset the scrolling flag after user stops scrolling."""
        self._is_scrolling = False
    
    def __init__(self, project: OptimizationProject):
        try:
            super().__init__()
            self.project = project
            # Lazy import - only import when OptimizationDashboard is actually instantiated
            from tui.optimization.optimizer import FileBasedOptimizer
            self.optimizer = FileBasedOptimizer(project)
            self.auto_run = False
            self.current_tab = "objective"  # Track current tab
            self._best_param_set_key = None  # (param_set_id, image_path) for best parameter set
            self._row_key_to_index = {}  # Cache mapping from (param_set_id, image_path) to row index
            self._table_rebuild_pending = False  # Flag to batch table rebuilds
            self._last_check_time = {}  # Track last check time per parameter set for rate limiting
            self._batch_check_index = 0  # Track which batch of parameter sets to check (for batch processing)
            self._is_scrolling = False  # Track if user is actively scrolling
            self._scroll_timer = None  # Timer to reset scroll flag
        except Exception as e:
            raise
        
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Grid(classes="dashboard-grid"):
            # --- Left Column: Method Config + Status (stacked) ---
            with Vertical(classes="left-column"):
                # Method Configuration Panel (Top)
                with Vertical(classes="method-config-panel"):
                    yield Label("Optimization Methods", classes="section-title")
                    yield Label("GP Kernel Type:")
                    yield Select(
                        [("RBF", "rbf"), ("Matérn", "matern"), ("Rational Quadratic", "rational_quadratic"), ("Exp Sine Squared", "exp_sine_squared")],
                        value=self.project.gp_kernel_type,
                        id="select-gp-kernel"
                    )
                    yield Label("Acquisition Function:")
                    yield Select(
                        [("UCB", "ucb"), ("Expected Improvement", "ei"), ("Probability of Improvement", "pi"), ("LCB", "lcb")],
                        value=self.project.acquisition_method,
                        id="select-acquisition"
                    )
                    yield Label("Pareto Algorithm:")
                    yield Select(
                        [("Dominance", "dominance"), ("NSGA-II", "nsga2"), ("NSGA-III", "nsga3")],
                        value=self.project.pareto_algorithm,
                        id="select-pareto"
                    )
                    with Horizontal(classes="method-config-actions"):
                        yield Button("Apply", id="btn-apply-methods", variant="primary")
                        yield Button("Reset to Defaults", id="btn-reset-methods", variant="default")
                
                # History Panel (Bottom) - Replaces status panel
                with Vertical(classes="history-panel"):
                    yield Label("Iteration History", classes="section-title")
                    with ScrollableContainer(id="table-scroll-container"):
                        yield DataTable(id="history-table")
                    
            # --- Right Column: Plot Panel (full height) ---
            with Vertical(classes="plot-panel"):
                # Tab buttons
                with Horizontal():
                    yield Button("▶ Objective Functions", id="tab-objective", classes="tab-btn--selected")
                    yield Button("  Pareto Front", id="tab-pareto")
                
                # Scrollable container for plots
                yield ScrollableContainer(id="plot-scroll-container")
                
        # --- Controls ---
        with Horizontal(classes="control-bar"):
            with Horizontal(classes="control-bar-left"):
                yield Button("Calculate", id="btn-calculate", variant="success")
                yield Button("Clear", id="btn-clear", variant="default")
                # yield Static("", id="calc-status")
                
            with Horizontal(classes="control-bar-right"):
                yield Static("Method:", id="method-label")
                yield Select(
                    [("Acquisition", "acquisition"), ("Pareto", "pareto"), ("Hybrid", "hybrid")],
                    value="acquisition",
                    id="select-suggestion-method",
                    allow_blank=False
                )
                yield Button("Suggest", id="btn-suggest", variant="primary")
                yield Button("Refresh", id="btn-refresh", variant="primary")
                yield Button("Auto", id="btn-auto", variant="warning")
                yield Button("Stop", id="btn-stop", variant="error")
                yield Button("Back", id="btn-back", variant="default")
            
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the dashboard with existing data."""
        import time
        mount_start = time.time()
        try:
            table = self.query_one("#history-table", DataTable)
            table_start = time.time()
            table.add_columns(
                "Image", "Param Set", 
                "Diameter", "Min Size", "CellProb", "Flow",
                "Status", 
                "Cells", "Mean Area",
                "Solidity", "Circular", "Euler%", "FCR", 
                "CV", "GeoCV", "LogNorm p", "Eccent", "Score"
            )
            columns_done = time.time()
            
            # Load existing parameter sets (fast - just display what we have)
            # Only show parameter sets from included config files
            history_start = time.time()
            self._row_key_to_index.clear()  # Clear cache on mount
            for ps in self.project.get_filtered_parameter_sets():
                self._add_row_to_table(table, ps, update_cache=True)
            history_done = time.time()
            
            # Update best parameter set after loading table
            self._update_best_parameter_set()
                    
            # Initialize plots for the current tab
            if self.current_tab == "objective":
                self.call_after_refresh(self.update_objective_function_plots)
            
            update_done = time.time()
            
            # Defer config file scanning to after UI is rendered (don't block initialization)
            # This scans config files and adds any new parameter sets found
            self.set_timer(0.5, self._scan_config_files_async)
            
            # Note: check_results is NOT called automatically on mount
            # User must click "Calculate" or "Refresh" to trigger calculations
        except Exception as e:
            raise

    def _add_row_to_table(self, table: DataTable, ps: ParameterSetEntry, update_cache: bool = True):
        # Determine status (cache this to avoid repeated file system checks)
        if ps.result is None:
            # Check if masks exist - if they do, mark as "Ready" for calculation
            # If not, mark as "Pending" (waiting for masks to be generated)
            if self.optimizer.has_output_files(ps):
                status = "Ready"  # Masks exist, ready for calculation
            else:
                status = "Pending"  # No masks yet, waiting
        elif self._has_default_quality_metrics(ps.result):
            status = "Ready"  # Has result but needs recalculation
        else:
            status = "Scored"  # Has valid result
        
        # Get image name and param_set_id for display
        # Use image_id if available, otherwise derive from image_path
        if ps.image_id:
            image_display = ps.image_id
        elif ps.image_path:
            image_display = os.path.basename(ps.image_path)
            # Remove .ome extension if present
            if image_display.endswith('.ome'):
                image_display = image_display[:-4]
        else:
            image_display = "..."
        
        param_set_id = ps.param_set_id or "..."
        row_key = (param_set_id, ps.image_path or "")
        
        # Check if this is the best parameter set
        is_best = (
            self._best_param_set_key is not None and
            self._best_param_set_key[0] == ps.param_set_id and
            self._best_param_set_key[1] == ps.image_path
        )
        
        # Add star indicator to image name if best
        display_image_name = f"★ {image_display}" if is_best else image_display
        
        # Format metric values
        if ps.result and not self._has_default_quality_metrics(ps.result):
            n_cells = str(ps.result.num_cells)
            mean_area = f"{ps.result.mean_area:.2f}"
            solidity = f"{ps.result.mean_solidity:.2f}"
            circular = f"{ps.result.mean_circularity:.2f}"
            euler_pct = f"{ps.result.euler_integrity_pct:.1f}%"
            fcr = f"{ps.result.fcr:.2f}" if ps.result.fcr >= 0 else "N/A"
            cv = f"{ps.result.cv_area:.2f}"
            geocv = f"{ps.result.geometric_cv:.2f}"
            lognorm_p = f"{ps.result.lognormal_pvalue:.3f}" if ps.result.lognormal_pvalue >= 0.001 else f"{ps.result.lognormal_pvalue:.2e}"
            eccent = f"{ps.result.mean_eccentricity:.2f}"
            score = f"{ps.result.score:.4f}"
        else:
            n_cells = ""
            mean_area = ""
            solidity = ""
            circular = ""
            euler_pct = ""
            fcr = ""
            cv = ""
            geocv = ""
            lognorm_p = ""
            eccent = ""
            score = ""
        
        row_key_str = f"{ps.param_set_id}:{ps.image_path}" if ps.param_set_id and ps.image_path else None
        row_key_obj = table.add_row(
            display_image_name,
            param_set_id,
            f"{ps.parameters.get('diameter', 0):.1f}",
            f"{ps.parameters.get('min_size', 0)}",
            f"{ps.parameters.get('cellprob_threshold', 0):.2f}",
            f"{ps.parameters.get('flow_threshold', 0):.2f}",
            status,
            n_cells,
            mean_area,
            solidity,
            circular,
            euler_pct,
            fcr,
            cv,
            geocv,
            lognorm_p,
            eccent,
            score,
            key=row_key_str
        )
        
        # Convert RowKey to integer index for storage and comparisons
        # Since we just added the row, it's at index row_count - 1
        # RowKey can be used directly in update_cell_at, but for comparisons we need int
        row_idx = table.row_count - 1
        
        # Update cache with integer index
        if update_cache and row_key_str:
            self._row_key_to_index[row_key] = row_idx


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            # Navigate back to project dashboard
            from tui.screens.project_dashboard import ProjectDashboard
            self.app.switch_screen(ProjectDashboard(self.project))
            
        elif event.button.id == "btn-calculate":
            # #region agent log
            import json
            try:
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "optimization_dashboard.py:267", "message": "Button clicked - calling calculate_stats_for_ready", "data": {"button_id": "btn-calculate"}, "timestamp": __import__("time").time() * 1000}) + "\n")
            except: pass
            # #endregion
            self.calculate_stats_for_ready()
            
        elif event.button.id == "btn-clear":
            self.clear_all_results()
            
        elif event.button.id == "btn-stop":
            self.auto_run = False
            self.project.status.state = "COMPLETED"
            self.project.save()
            self.notify("Optimization stopped.")
            
        elif event.button.id == "btn-refresh":
            self.refresh_iterations()
            # Also check for completed results
            self.check_results()
            
        elif event.button.id == "btn-auto":
            self.toggle_auto_run()
        
        elif event.button.id == "btn-suggest":
            # Get selected method from dropdown
            method_select = self.query_one("#select-suggestion-method", Select)
            method = method_select.value if method_select.value else "acquisition"
            self.show_suggestion_dialog(method=method)
        
        elif event.button.id == "tab-objective":
            self._switch_tab("objective")
            
        elif event.button.id == "tab-pareto":
            self._switch_tab("pareto")
        
        elif event.button.id == "btn-apply-methods":
            self._apply_method_selections()
        
        elif event.button.id == "btn-reset-methods":
            self._reset_method_selections()
        
        elif event.button.id and event.button.id.startswith("direction-btn-"):
            # Handle direction button clicks
            obj_name = event.button.id.replace("direction-btn-", "")
            self._toggle_objective_direction(obj_name)

    def toggle_auto_run(self):
        self.auto_run = not self.auto_run
        btn = self.query_one("#btn-auto", Button)
        if self.auto_run:
            btn.label = "Pause Auto-Run"
            btn.variant = "error"
            # Don't auto-refresh immediately, just enable auto-checking
        else:
            btn.label = "Auto-Run"
            btn.variant = "warning"

    def _scan_config_files_async(self):
        """Scan config files asynchronously after UI is rendered."""
        import time
        scan_start = time.time()
        try:
            # Get count before scanning
            old_count = len(self.project.parameter_sets)
            
            # Scan config files to get current parameter sets (this will add new ones via get_or_create_parameter_set)
            scanned_parameter_sets = self.optimizer.scan_config_files_for_parameter_sets()
            scan_done = time.time()
            
            # Check if new parameter sets were added
            new_count = len(self.project.parameter_sets) - old_count
            if new_count > 0:
                # New parameter sets were added, save and update table
                self.project.save()
                logger.info(f"Added {new_count} new parameter set(s) from config files, total: {len(self.project.parameter_sets)}")
                
                # Defer table rebuild to avoid blocking UI
                def rebuild_table():
                    if not self._is_scrolling:  # Skip if user is scrolling
                        table = self.query_one("#history-table", DataTable)
                        # Rebuild table with filtered sets only
                        filtered_sets = self.project.get_filtered_parameter_sets()
                        table.clear()
                        self._row_key_to_index.clear()  # Clear cache when rebuilding
                        for ps in filtered_sets:
                            self._add_row_to_table(table, ps, update_cache=True)
                
                self.call_after_refresh(rebuild_table)
                
        except Exception as e:
            logger.error(f"Error scanning config files: {e}")
    
    def check_results(self):
        """Check for updates on pending iterations."""
        import time
        # Skip updates if user is actively scrolling to prevent glitches
        if self._is_scrolling:
            return
        
        check_start = time.time()
        try:
            table = self.query_one("#history-table", DataTable)
            updated = False
            latest_completed = False
            
            # Use filtered parameter sets (only from included config files)
            filtered_sets = self.project.get_filtered_parameter_sets()
            pending_count = 0
            
            # Rebuild row mapping cache if table size changed (more efficient than rebuilding every time)
            if len(self._row_key_to_index) != table.row_count:
                self._row_key_to_index.clear()
                for row_idx in range(table.row_count):
                    try:
                        image_cell = table.get_cell_at((row_idx, 0))
                        param_cell = table.get_cell_at((row_idx, 1))
                        # Remove star prefix if present
                        image_name = image_cell.value.replace("★ ", "") if image_cell.value else "..."
                        param_set_id = param_cell.value if param_cell.value else "..."
                        # Try to get image_path from parameter set to match row_key format
                        # For now, use image_name as fallback (less precise but works)
                        row_key = (param_set_id, image_name)
                        self._row_key_to_index[row_key] = row_idx
                    except Exception:
                        continue
            
            # Batch processing: Only check a subset of pending parameter sets per cycle
            # Filter to get only pending sets (those without results)
            pending_sets = [ps for ps in filtered_sets if ps.result is None]
            pending_count = len(pending_sets)
            
            # Process only a batch at a time (e.g., 10 sets per cycle) to avoid overwhelming the system
            batch_size = 10
            if pending_sets:
                # Calculate which batch to process (rotate through all pending sets)
                start_idx = self._batch_check_index % len(pending_sets)
                end_idx = min(start_idx + batch_size, len(pending_sets))
                
                # Get the batch to process
                batch_to_check = pending_sets[start_idx:end_idx]
                
                # Update batch index for next cycle (wrap around if needed)
                self._batch_check_index = end_idx if end_idx < len(pending_sets) else 0
            else:
                batch_to_check = []
            
            # Process only the batch
            for i, param_set in enumerate(batch_to_check):
                # FIRST: Lightweight check if files exist (avoids expensive calculation when no output yet)
                has_files = self.optimizer.has_output_files(param_set)
                
                # Update status to "Ready" if masks exist but no result yet
                if has_files and param_set.result is None:
                    param_set_id = param_set.param_set_id or "..."
                    row_key = (param_set_id, param_set.image_path or "")
                    row_idx = self._row_key_to_index.get(row_key)
                    
                    if row_idx is not None and row_idx < table.row_count:
                        # Update status to "Ready" since masks exist
                        table.update_cell_at((row_idx, 6), "Ready")
                
                if not has_files:
                    continue  # Skip expensive calculation if no files yet
                
                # Rate limiting: Only check if enough time has passed since last check (30 seconds minimum)
                param_set_key = (param_set.param_set_id or "", param_set.image_path or "")
                current_time = time.time()
                last_check = self._last_check_time.get(param_set_key, 0)
                min_check_interval = 30.0  # Minimum seconds between checks
                
                if current_time - last_check < min_check_interval:
                    continue  # Skip if checked too recently
                
                # Update last check time
                self._last_check_time[param_set_key] = current_time
                
                # THEN: Only do expensive calculation if files exist and rate limit allows
                # Check status
                status_check_start = time.time()
                result = self.optimizer.check_parameter_set_status(param_set)
                status_check_done = time.time()
                if result:
                    self.project.update_parameter_set_result(
                        param_set_id=param_set.param_set_id,
                        image_path=param_set.image_path or "",
                        result=result,
                        config_file_path=param_set.config_file_path
                    )
                    updated = True
                    
                    # Update table - find row index using cached mapping
                    param_set_id = param_set.param_set_id or "..."
                    row_key = (param_set_id, param_set.image_path or "")
                    row_idx = self._row_key_to_index.get(row_key)
                    
                    if row_idx is not None and row_idx < table.row_count:
                        # Determine status
                        has_defaults = self._has_default_quality_metrics(result)
                        status_text = "Scored" if not has_defaults else "Ready"
                        
                        # Batch cell updates to reduce redraws
                        updates = []
                        updates.append(((row_idx, 6), status_text))  # Status column (6)
                        if not has_defaults:
                            # Column indices: Config(0), Param Set(1), Diameter(2), Min Size(3), CellProb(4), Flow(5), 
                            # Status(6), Cells(7), Mean Area(8), Solidity(9), Circular(10), Euler%(11), FCR(12),
                            # CV(13), GeoCV(14), LogNorm p(15), Eccent(16), Score(17)
                            updates.append(((row_idx, 7), str(result.num_cells)))  # Cells column
                            updates.append(((row_idx, 8), f"{result.mean_area:.2f}"))  # Mean Area column
                            updates.append(((row_idx, 9), f"{result.mean_solidity:.2f}"))  # Solidity
                            updates.append(((row_idx, 10), f"{result.mean_circularity:.2f}"))  # Circular
                            updates.append(((row_idx, 11), f"{result.euler_integrity_pct:.1f}%"))  # Euler%
                            fcr_val = f"{result.fcr:.2f}" if result.fcr >= 0 else "N/A"
                            updates.append(((row_idx, 12), fcr_val))  # FCR
                            updates.append(((row_idx, 13), f"{result.cv_area:.2f}"))  # CV
                            updates.append(((row_idx, 14), f"{result.geometric_cv:.2f}"))  # GeoCV
                            lognorm_p = f"{result.lognormal_pvalue:.3f}" if result.lognormal_pvalue >= 0.001 else f"{result.lognormal_pvalue:.2e}"
                            updates.append(((row_idx, 15), lognorm_p))  # LogNorm p
                            updates.append(((row_idx, 16), f"{result.mean_eccentricity:.2f}"))  # Eccent
                            updates.append(((row_idx, 17), f"{result.score:.4f}"))  # Score column
                        
                        # Defer table updates to avoid blocking UI during scroll
                        # Use call_after_refresh to batch updates and prevent glitches
                        def apply_updates():
                            if not self._is_scrolling:  # Double-check we're not scrolling
                                for (r, c), value in updates:
                                    try:
                                        table.update_cell_at((r, c), value)
                                    except Exception:
                                        pass  # Ignore errors if table was modified
                        
                        self.call_after_refresh(apply_updates)
                         
                    # If this was the last parameter set in the batch, mark as recently completed
                    if i == len(batch_to_check) - 1:
                        latest_completed = True

            check_done = time.time()

            if updated:
                self.project.save()
                
                # Update best parameter set after results change (defer to avoid blocking)
                if not self._is_scrolling:
                    self.call_after_refresh(self._update_best_parameter_set)
                
                # Auto-run logic: if the latest iteration just completed, refresh to check for new iterations
                if self.auto_run and latest_completed and self.project.status.state != "COMPLETED":
                    self.refresh_iterations()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f'Error: {e}')
            # Don't raise - just log the error

    @work(exclusive=True)
    async def refresh_iterations(self):
        """Refresh parameter sets by re-scanning config files."""
        self.project.status.state = "RUNNING"
        
        try:
            # Get count before scanning
            old_count = len(self.project.parameter_sets)
            
            # Scan config files for parameter sets (this will add new ones via get_or_create_parameter_set)
            scanned_parameter_sets = self.optimizer.scan_config_files_for_parameter_sets()
            
            new_count = len(self.project.parameter_sets)
            added_count = new_count - old_count
            
            if added_count > 0:
                self.project.save()
                
                # Defer table rebuild to avoid blocking UI
                def rebuild_table_async():
                    if not self._is_scrolling:  # Skip if user is scrolling
                        table = self.query_one("#history-table", DataTable)
                        # Rebuild table with filtered sets only
                        filtered_sets = self.project.get_filtered_parameter_sets()
                        table.clear()
                        self._row_key_to_index.clear()  # Clear cache when rebuilding
                        for param_set in filtered_sets:
                            self._add_row_to_table(table, param_set, update_cache=True)
                        # Scroll to end only once after all rows are added
                        table.scroll_end()
                        
                        # Update best parameter set after refresh
                        self._update_best_parameter_set()
                
                self.call_after_refresh(rebuild_table_async)
                
                self.notify(f"Added {added_count} new parameter set(s) from config files.")
            else:
                self.notify("No new parameter sets found in config files.")
                
        except Exception as e:
            self.notify(f"Error refreshing parameter sets: {e}", severity="error")
            logger.error(f"Error refreshing parameter sets: {e}")
        
        self.project.status.state = "SETUP"
    
    def _is_ready_for_scoring(self, param_set: ParameterSetEntry) -> bool:
        """Check if a parameter set is ready for scoring.
        
        A parameter set is ready if:
        1. It has no result yet (masks exist but not calculated), OR
        2. It has a result but with default/zero quality metrics (needs recalculation)
        
        Args:
            param_set: ParameterSetEntry to check
            
        Returns:
            True if ready for scoring, False otherwise
        """
        # Check if masks exist for this parameter set
        has_masks = self.optimizer.has_output_files(param_set)
        
        if not has_masks:
            return False  # No masks yet, not ready
        
        # If no result, it's ready
        if param_set.result is None:
            return True
        
        # If result exists but has default metrics, it needs recalculation
        if self._has_default_quality_metrics(param_set.result):
            return True
        
        # Has result with valid metrics, already scored
        return False
    
    def _has_default_quality_metrics(self, result: OptimizationResult) -> bool:
        """Check if a result has default/zero quality metrics.
        
        This indicates the result was migrated from old format or calculation failed.
        
        Args:
            result: OptimizationResult to check
            
        Returns:
            True if result has default metrics (all quality metrics are 0.0)
        """
        return self.project._result_has_default_metrics(result)
    
    def _update_best_parameter_set(self, show_notification_on_change: bool = True):
        """Update the best parameter set based on current optimization method.
        
        Args:
            show_notification_on_change: If True, show notification when best changes
        """
        # Skip if user is actively scrolling to prevent glitches
        if self._is_scrolling:
            return
        
        try:
            from tui.optimization.optimizer import identify_best_parameter_set
            
            # Determine method from current settings
            # Use the same method as suggestion, or fallback to acquisition
            method = "acquisition"  # Default
            try:
                method_select = self.query_one("#select-suggestion-method", Select)
                if method_select and method_select.value:
                    method = method_select.value
            except Exception:
                # Widget might not be available yet, use default
                pass
            
            best_key = identify_best_parameter_set(self.project, method=method)
            old_best = self._best_param_set_key
            self._best_param_set_key = best_key
            
            # Only rebuild table if best changed (to avoid expensive rebuilds)
            if old_best != best_key:
                try:
                    table = self.query_one("#history-table", DataTable)
                    filtered_sets = self.project.get_filtered_parameter_sets()
                    
                    # Update star indicators without full rebuild - just update affected rows
                    if old_best:
                        old_param_set_id = old_best[0] or "..."
                        old_image_path = old_best[1] or ""
                        old_row_key = (old_param_set_id, old_image_path)
                        old_row_idx = self._row_key_to_index.get(old_row_key)
                        if old_row_idx is not None and old_row_idx < table.row_count:
                            # Remove star from old best
                            old_image_cell = table.get_cell_at((old_row_idx, 0))
                            if old_image_cell.value.startswith("★ "):
                                table.update_cell_at((old_row_idx, 0), old_image_cell.value.replace("★ ", ""))
                    
                    if best_key:
                        new_param_set_id = best_key[0] or "..."
                        new_image_path = best_key[1] or ""
                        new_row_key = (new_param_set_id, new_image_path)
                        new_row_idx = self._row_key_to_index.get(new_row_key)
                        if new_row_idx is not None and new_row_idx < table.row_count:
                            # Add star to new best
                            new_image_cell = table.get_cell_at((new_row_idx, 0))
                            if not new_image_cell.value.startswith("★ "):
                                table.update_cell_at((new_row_idx, 0), f"★ {new_image_cell.value}")
                        
                        if show_notification_on_change:
                            image_name = os.path.basename(new_image_path) if new_image_path else "..."
                            self.notify(f"Best parameter set updated: {new_param_set_id} / {image_name}", severity="information")
                        logger.debug(f"Best parameter set: {new_param_set_id} / {new_image_path} (method: {method})")
                    else:
                        logger.debug("No best parameter set found (no scored results?)")
                except Exception as e:
                    logger.error(f"Could not update table for best parameter set: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error updating best parameter set: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Don't raise - just log the error
    
    @work(exclusive=True, thread=True)
    def calculate_stats_for_ready(self):
        """Calculate statistics for all parameter sets that are ready for scoring."""
        # #region agent log
        import json, time, threading
        try:
            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "optimization_dashboard.py:680", "message": "Method entry - worker starting", "data": {"thread_id": threading.current_thread().ident, "thread_name": threading.current_thread().name}, "timestamp": time.time() * 1000}) + "\n")
        except: pass
        # #endregion
        try:
            logger.info("calculate_stats_for_ready: Starting calculation")
            
            # #region agent log
            try:
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "optimization_dashboard.py:685", "message": "Inside try block - worker thread confirmed", "data": {}, "timestamp": time.time() * 1000}) + "\n")
            except: pass
            # #endregion
            
            # Notify that we're starting (on main thread)
            def notify_starting():
                self.notify("Starting calculation...", severity="info")
            # #region agent log
            try:
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "optimization_dashboard.py:691", "message": "Before call_from_thread for notify_starting", "data": {}, "timestamp": time.time() * 1000}) + "\n")
            except: pass
            # #endregion
            self.app.call_from_thread(notify_starting)
            
            # Get filtered parameter sets (only from included config files)
            filtered_sets = self.project.get_filtered_parameter_sets()
            logger.info(f"calculate_stats_for_ready: Found {len(filtered_sets)} filtered parameter sets")
            # #region agent log
            try:
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "optimization_dashboard.py:696", "message": "Filtered sets retrieved", "data": {"count": len(filtered_sets)}, "timestamp": time.time() * 1000}) + "\n")
            except: pass
            # #endregion
            
            # Find parameter sets that are ready for scoring
            ready_sets = []
            for ps in filtered_sets:
                try:
                    is_ready = self._is_ready_for_scoring(ps)
                    if is_ready:
                        ready_sets.append(ps)
                    logger.debug(f"Parameter set {ps.param_set_id} / {ps.image_path}: has_masks={self.optimizer.has_output_files(ps)}, result={ps.result is not None}, ready={is_ready}")
                except Exception as e:
                    logger.error(f"Error checking if parameter set {ps.param_set_id} is ready: {e}", exc_info=True)
            
            logger.info(f"Found {len(ready_sets)} ready parameter sets out of {len(filtered_sets)} total")
            # #region agent log
            try:
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "optimization_dashboard.py:710", "message": "Ready sets count", "data": {"ready_count": len(ready_sets), "total_count": len(filtered_sets)}, "timestamp": time.time() * 1000}) + "\n")
            except: pass
            # #endregion
            
            if not ready_sets:
                def notify_no_ready():
                    self.notify(f"No parameter sets ready for calculation. Checked {len(filtered_sets)} sets. Check that masks exist and status is 'Ready'.", severity="info")
                # #region agent log
                try:
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "optimization_dashboard.py:714", "message": "No ready sets - returning early", "data": {}, "timestamp": time.time() * 1000}) + "\n")
                except: pass
                # #endregion
                self.app.call_from_thread(notify_no_ready)
                return
            
            # Update status (must use call_from_thread since we're in a worker thread)
            def update_status():
                try:
                    from textual.widgets import Static
                    calc_status = self.query_one("#calc-status", Static)
                    calc_status.update(f"Calculating stats for {len(ready_sets)} set(s)...")
                    # #region agent log
                    try:
                        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "D", "location": "optimization_dashboard.py:722", "message": "Status widget updated successfully", "data": {"ready_count": len(ready_sets)}, "timestamp": time.time() * 1000}) + "\n")
                    except: pass
                    # #endregion
                except Exception as e:
                    # #region agent log
                    try:
                        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "D", "location": "optimization_dashboard.py:725", "message": "Status widget update failed", "data": {"error": str(e)}, "timestamp": time.time() * 1000}) + "\n")
                    except: pass
                    # #endregion
                    pass  # Widget might not exist
            # #region agent log
            try:
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "optimization_dashboard.py:732", "message": "Before call_from_thread for update_status", "data": {"ready_count": len(ready_sets)}, "timestamp": time.time() * 1000}) + "\n")
            except: pass
            # #endregion
            self.app.call_from_thread(update_status)
            
            updated_count = 0
            error_count = 0
            
            # Braille spinner frames for status updates
            spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spinner_idx = 0
            
            # #region agent log
            try:
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "optimization_dashboard.py:740", "message": "Starting loop over ready_sets", "data": {"ready_count": len(ready_sets)}, "timestamp": time.time() * 1000}) + "\n")
            except: pass
            # #endregion
            
            # Use ThreadPoolExecutor to run heavy calculations in separate threads
            # This allows the worker thread to yield and update UI
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time
            
            # Process calculations with a limited number of concurrent workers
            max_workers = 2  # Limit concurrent calculations to avoid overwhelming system
            
            def calculate_single(param_set_with_idx):
                """Calculate stats for a single parameter set - runs in thread pool."""
                param_set, actual_idx = param_set_with_idx
                try:
                    result = self.optimizer.check_parameter_set_status(param_set)
                    return (param_set, actual_idx, result, None)
                except Exception as e:
                    return (param_set, actual_idx, None, e)
            
            # Track active calculations for spinner animation
            active_calculations = {}  # Maps row_idx_int to (future, param_set)
            import threading
            stop_spinner_animation = threading.Event()
            
            # Helper function to find row index for a parameter set (returns integer index)
            def find_row_idx_for_param_set(param_set):
                import os
                row_key = (param_set.param_set_id or "", param_set.image_path or "")
                row_idx = self._row_key_to_index.get(row_key)
                
                # Fallback: find by matching all fields in filtered_sets
                if row_idx is None:
                    for i, ps in enumerate(filtered_sets):
                        ps_image = os.path.normpath(ps.image_path) if ps.image_path else None
                        param_image = os.path.normpath(param_set.image_path) if param_set.image_path else None
                        
                        if (ps.param_set_id == param_set.param_set_id and 
                            ps_image == param_image and
                            ps.config_file_path == param_set.config_file_path):
                            row_idx = i
                            self._row_key_to_index[row_key] = row_idx
                            break
                
                # Return integer index (cache should already have integers)
                return row_idx if isinstance(row_idx, int) else None
            
            # Submit all calculations to thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # First, set "Calculating" for all rows that will be calculated
                def set_all_calculating():
                    table = self.query_one("#history-table", DataTable)
                    for param_set in ready_sets:
                        row_idx_int = find_row_idx_for_param_set(param_set)
                        if row_idx_int is not None and isinstance(row_idx_int, int) and row_idx_int < table.row_count:
                            table.update_cell_at((row_idx_int, 6), "Calculating")
                self.app.call_from_thread(set_all_calculating)
                
                # Submit all calculations
                futures = {}
                for idx, param_set in enumerate(ready_sets):
                    future = executor.submit(calculate_single, (param_set, idx))
                    futures[future] = (param_set, idx)
                    
                    # Find row index and track for spinner animation
                    row_idx_int = find_row_idx_for_param_set(param_set)
                    if row_idx_int is not None:
                        # Track this calculation for spinner animation
                        active_calculations[row_idx_int] = (future, param_set)
                        # Immediately set spinner for this row
                        spinner_char = spinner_frames[0]
                        def update_spinner_immediate(row_idx=row_idx_int, spinner=spinner_char):
                            table = self.query_one("#history-table", DataTable)
                            if isinstance(row_idx, int) and row_idx < table.row_count:
                                table.update_cell_at((row_idx, 6), spinner)
                        self.app.call_from_thread(update_spinner_immediate)
                
                # Animate spinner for active calculations in a separate loop
                def animate_spinners():
                    """Continuously update spinner animation for active calculations."""
                    # Small delay to ensure calculations have started
                    time.sleep(0.1)
                    frame_idx = 0
                    while not stop_spinner_animation.is_set():
                        if not active_calculations:
                            time.sleep(0.1)
                            continue
                        
                        frame_idx = (frame_idx + 1) % len(spinner_frames)
                        spinner_char = spinner_frames[frame_idx]
                        
                        # Update all active calculations with spinner only
                        def update_all_spinners():
                            table = self.query_one("#history-table", DataTable)
                            for row_idx_int, (future, param_set) in list(active_calculations.items()):
                                if isinstance(row_idx_int, int) and row_idx_int < table.row_count:
                                    # Only update if still running - show ONLY spinner character
                                    if not future.done():
                                        table.update_cell_at((row_idx_int, 6), spinner_char)
                        self.app.call_from_thread(update_all_spinners)
                        time.sleep(0.2)  # Update every 200ms
                
                # Start spinner animation thread
                spinner_thread = threading.Thread(target=animate_spinners, daemon=True)
                spinner_thread.start()
                
                # Process results as they complete
                for future in as_completed(futures):
                    param_set, actual_idx, result, error = future.result()
                    
                    # Remove from active calculations
                    row_idx_int = find_row_idx_for_param_set(param_set)
                    if row_idx_int is not None and row_idx_int in active_calculations:
                        del active_calculations[row_idx_int]
                    
                    # Process result as it completes
                    # #region agent log
                    try:
                        import json, time
                        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "F", "location": "optimization_dashboard.py:800", "message": "Calculation completed", "data": {"iteration": actual_idx, "param_set_id": param_set.param_set_id, "has_result": result is not None, "has_error": error is not None}, "timestamp": time.time() * 1000}) + "\n")
                    except: pass
                    # #endregion
                    
                    try:
                        if row_idx_int is None:
                            row_idx_int = find_row_idx_for_param_set(param_set)
                        
                        if row_idx_int is None:
                            logger.warning(f"Could not find row for param_set_id={param_set.param_set_id}, image_path={param_set.image_path}")
                            continue
                        
                        # Small yield to allow UI to update
                        time.sleep(0.05)
                        
                        # Handle error case
                        if error:
                            error_count += 1
                            logger.error(f"Error calculating stats for {param_set.param_set_id}: {error}", exc_info=True)
                            def update_table_error():
                                table = self.query_one("#history-table", DataTable)
                                if isinstance(row_idx_int, int) and row_idx_int < table.row_count:
                                    table.update_cell_at((row_idx_int, 6), "Error")
                            self.app.call_from_thread(update_table_error)
                            continue
                        
                        # Handle successful result
                        if result:
                            self.project.update_parameter_set_result(
                                param_set_id=param_set.param_set_id,
                                image_path=param_set.image_path or "",
                                result=result,
                                config_file_path=param_set.config_file_path
                            )
                            updated_count += 1
                            
                            # Update table - clear spinner and show results (use call_from_thread)
                            def update_table_scored():
                                table = self.query_one("#history-table", DataTable)
                                if isinstance(row_idx_int, int) and row_idx_int < table.row_count:
                                    # Only mark as "Scored" if result doesn't have default values
                                    has_defaults = self._has_default_quality_metrics(result)
                                    status_text = "Scored" if not has_defaults else "Ready"
                                    table.update_cell_at((row_idx_int, 6), status_text)  # Status column
                                    
                                    # Update ALL metric cells if result doesn't have default values
                                    # Use same formatting as _add_row_to_table
                                    if not has_defaults:
                                        table.update_cell_at((row_idx_int, 7), str(result.num_cells))  # Cells
                                        table.update_cell_at((row_idx_int, 8), f"{result.mean_area:.2f}")  # Mean Area
                                        table.update_cell_at((row_idx_int, 9), f"{result.mean_solidity:.2f}")  # Solidity
                                        table.update_cell_at((row_idx_int, 10), f"{result.mean_circularity:.2f}")  # Circular
                                        table.update_cell_at((row_idx_int, 11), f"{result.euler_integrity_pct:.1f}%")  # Euler%
                                        fcr_str = f"{result.fcr:.2f}" if result.fcr >= 0 else "N/A"
                                        table.update_cell_at((row_idx_int, 12), fcr_str)  # FCR
                                        table.update_cell_at((row_idx_int, 13), f"{result.cv_area:.2f}")  # CV
                                        table.update_cell_at((row_idx_int, 14), f"{result.geometric_cv:.2f}")  # GeoCV
                                        lognorm_str = f"{result.lognormal_pvalue:.3f}" if result.lognormal_pvalue >= 0.001 else f"{result.lognormal_pvalue:.2e}"
                                        table.update_cell_at((row_idx_int, 15), lognorm_str)  # LogNorm p
                                        table.update_cell_at((row_idx_int, 16), f"{result.mean_eccentricity:.2f}")  # Eccent
                                        table.update_cell_at((row_idx_int, 17), f"{result.score:.4f}")  # Score
                            
                            self.app.call_from_thread(update_table_scored)
                        else:
                            error_count += 1
                            def update_table_error():
                                table = self.query_one("#history-table", DataTable)
                                if isinstance(row_idx_int, int) and row_idx_int < table.row_count:
                                    table.update_cell_at((row_idx_int, 6), "Error")
                            self.app.call_from_thread(update_table_error)
                    
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error processing result for {param_set.param_set_id}: {e}", exc_info=True)
                
                # Stop spinner animation when all calculations are done
                stop_spinner_animation.set()
                # Wait a bit for the spinner thread to finish
                time.sleep(0.3)
            
            # Update UI with final results (use call_from_thread)
            def update_final_ui():
                from textual.widgets import Static
                # Check if quality metrics were calculated - if not, warn about missing dependencies
                if updated_count > 0:
                    # Check if any results have default metrics (indicating missing dependencies)
                    filtered_sets = self.project.get_filtered_parameter_sets()
                    results_with_defaults = sum(1 for ps in filtered_sets 
                                              if ps.result is not None and self._has_default_quality_metrics(ps.result))
                    if results_with_defaults > 0:
                        # Check if scikit-image is missing
                        try:
                            import skimage
                        except ImportError:
                            self.notify(
                                "Quality metrics could not be calculated: scikit-image is missing. "
                                "Install with: pip install scikit-image scipy",
                                severity="warning"
                            )
                        else:
                            try:
                                import scipy
                            except ImportError:
                                self.notify(
                                    "Quality metrics could not be calculated: scipy is missing. "
                                    "Install with: pip install scipy",
                                    severity="warning"
                                )
                    try:
                        calc_status = self.query_one("#calc-status", Static)
                        if updated_count > 0:
                            calc_status.update(f"✓ Completed: {updated_count} set(s) calculated")
                            if error_count > 0:
                                self.notify(f"Calculated stats for {updated_count} set(s). {error_count} error(s).", severity="warning")
                            else:
                                self.notify(f"Successfully calculated stats for {updated_count} parameter set(s).", severity="success")
                            
                            # Update best parameter set after calculations
                            self._update_best_parameter_set()
                        else:
                            calc_status.update(f"✗ No stats calculated ({error_count} error(s))")
                    except Exception:
                        pass  # Widget might not exist
            
            self.app.call_from_thread(update_final_ui)
            
            # Save project
            if updated_count > 0:
                self.project.save()
                
        except Exception as e:
            logger.error(f"Error in calculate_stats_for_ready: {e}", exc_info=True)
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "optimization_dashboard.py:872", "message": "Exception caught in calculate_stats_for_ready", "data": {"error": str(e), "error_type": type(e).__name__}, "timestamp": time.time() * 1000}) + "\n")
            except: pass
            # #endregion
            def notify_error():
                self.notify(f"Error during calculation: {e}", severity="error")
            self.app.call_from_thread(notify_error)
    
    def clear_all_results(self):
        """Clear all stored results from parameter sets, allowing recalculation."""
        from textual.widgets import DataTable
        
        # Clear results from all parameter sets
        cleared_count = 0
        for ps in self.project.parameter_sets:
            if ps.result is not None:
                ps.result = None
                ps.last_updated = None
                cleared_count += 1
        
        # Reset best score tracking
        self.project.status.best_score = 0.0
        self.project.status.best_parameters = {}
        self.project.status.best_param_set_id = None
        self.project.status.best_config_file_path = None
        self._best_param_set_key = None
        
        # Save the project
        self.project.save()
        
        # Update the table to reflect cleared status
        try:
            table = self.query_one("#history-table", DataTable)
            filtered_sets = self.project.get_filtered_parameter_sets()
            
            # Update status column for all rows
            for i, ps in enumerate(filtered_sets):
                if i < table.row_count:
                    # Update status to "Pending" since results are cleared
                    table.update_cell_at((i, 6), "Pending")  # Status column (index 6)
                    
                    # Clear metric columns
                    table.update_cell_at((i, 7), "")  # Cells column
                    table.update_cell_at((i, 8), "")  # Mean Area column
                    table.update_cell_at((i, 9), "")  # Solidity
                    table.update_cell_at((i, 10), "")  # Circular
                    table.update_cell_at((i, 11), "")  # Euler%
                    table.update_cell_at((i, 12), "")  # FCR
                    table.update_cell_at((i, 13), "")  # CV
                    table.update_cell_at((i, 14), "")  # GeoCV
                    table.update_cell_at((i, 15), "")  # LogNorm p
                    table.update_cell_at((i, 16), "")  # Eccent
                    table.update_cell_at((i, 17), "")  # Score column
            
            # Update best parameter set display (will clear the star)
            self._update_best_parameter_set(show_notification_on_change=False)
            
        except Exception as e:
            logger.error(f"Error updating table after clearing results: {e}")
        
        # Show notification
        self.notify(f"Cleared results for {cleared_count} parameter set(s). Ready for recalculation.", severity="success")
        
        # Refresh plots since results are cleared
        if self.current_tab == "objective":
            self.call_after_refresh(self.update_objective_function_plots)
        elif self.current_tab == "pareto":
            self.call_after_refresh(self.update_pareto_front_plots)
    
    def _switch_tab(self, tab_name: str) -> None:
        """Switch the visible tab in the plot panel."""
        self.current_tab = tab_name
        
        # Update buttons
        btn_obj = self.query_one("#tab-objective", Button)
        btn_par = self.query_one("#tab-pareto", Button)
        
        if tab_name == "objective":
            btn_obj.add_class("tab-btn--selected")
            btn_par.remove_class("tab-btn--selected")
            btn_obj.label = "▶ Objective Functions"
            btn_par.label = "  Pareto Front"
            self.call_after_refresh(self.update_objective_function_plots)
        else:
            btn_par.add_class("tab-btn--selected")
            btn_obj.remove_class("tab-btn--selected")
            btn_par.label = "▶ Pareto Front"
            btn_obj.label = "  Objective Functions"
            self.call_after_refresh(self.update_pareto_front_plots)
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes - auto-apply when changed."""
        if event.select.id in ["select-gp-kernel", "select-acquisition", "select-pareto"]:
            if event.select.id == "select-gp-kernel":
                self.project.gp_kernel_type = event.value
            elif event.select.id == "select-acquisition":
                self.project.acquisition_method = event.value
            elif event.select.id == "select-pareto":
                self.project.pareto_algorithm = event.value
            self.project.save()
            # Refresh plots if on relevant tab
            if self.current_tab == "objective":
                self.call_after_refresh(self.update_objective_function_plots)
            elif self.current_tab == "pareto":
                self.call_after_refresh(self.update_pareto_front_plots)
    
    def _apply_method_selections(self) -> None:
        """Apply method selections from dropdowns."""
        gp_kernel = self.query_one("#select-gp-kernel", Select).value
        acquisition = self.query_one("#select-acquisition", Select).value
        pareto = self.query_one("#select-pareto", Select).value
        
        self.project.gp_kernel_type = gp_kernel
        self.project.acquisition_method = acquisition
        self.project.pareto_algorithm = pareto
        self.project.save()
        
        # Refresh plots
        if self.current_tab == "objective":
            self.call_after_refresh(self.update_objective_function_plots)
        elif self.current_tab == "pareto":
            self.call_after_refresh(self.update_pareto_front_plots)
        
        self.notify("Methods applied", severity="success")
    
    def _reset_method_selections(self) -> None:
        """Reset method selections to defaults."""
        self.project.gp_kernel_type = "rbf"
        self.project.acquisition_method = "ucb"
        self.project.pareto_algorithm = "dominance"
        self.project.save()
        
        # Update UI
        self.query_one("#select-gp-kernel", Select).value = "rbf"
        self.query_one("#select-acquisition", Select).value = "ucb"
        self.query_one("#select-pareto", Select).value = "dominance"
        
        # Refresh plots
        if self.current_tab == "objective":
            self.call_after_refresh(self.update_objective_function_plots)
        elif self.current_tab == "pareto":
            self.call_after_refresh(self.update_pareto_front_plots)
        
        self.notify("Methods reset to defaults", severity="information")
    
    def _toggle_objective_direction(self, obj_name: str) -> None:
        """Toggle the optimization direction for an objective."""
        # Get the current stored direction (don't use default for cycling)
        # The default is only used for initial display, not for cycling logic
        current = self.project.objective_directions.get(obj_name, None)
        
        # Cycle: True -> False -> None -> True
        # Always cycle based on what's actually stored, not the default
        if current is True:
            new_direction = False
        elif current is False:
            new_direction = None
        else:  # None (or not set)
            new_direction = True
        
        # Update project
        self.project.objective_directions[obj_name] = new_direction
        self.project.save()
        
        # Update button and border title directly (no plot refresh needed)
        
        # Update best parameter set since direction change affects selection
        # Store old best before update to check if it changed
        old_best_before_update = self._best_param_set_key
        self._update_best_parameter_set(show_notification_on_change=True)
        
        # Show notification about direction change if best didn't change
        # (if best changed, _update_best_parameter_set already showed notification)
        direction_text = "maximize" if new_direction is True else "minimize" if new_direction is False else "not used"
        if old_best_before_update == self._best_param_set_key:
            # Best didn't change, but show confirmation that direction was updated
            if self._best_param_set_key:
                param_set_id = self._best_param_set_key[0] or "..."
                image_path = self._best_param_set_key[1] or "..."
                image_name = os.path.basename(image_path) if image_path != "..." else "..."
                self.notify(f"Objective '{obj_name}' set to {direction_text}. Best unchanged: {param_set_id} / {image_name}", severity="information")
            else:
                self.notify(f"Objective '{obj_name}' set to {direction_text}. No best parameter set found.", severity="information")
        try:
            btn = self.query_one(f"#direction-btn-{obj_name}", Button)
            if new_direction is True:
                btn.label = "↑"
                btn.tooltip = "Maximize (click to change)"
            elif new_direction is False:
                btn.label = "↓"
                btn.tooltip = "Minimize (click to change)"
            else:
                btn.label = "○"
                btn.tooltip = "Not used in Pareto (click to change)"
            
            # Update border title on the same card
            card = btn.parent
            while card and not hasattr(card, 'border_title'):
                card = card.parent
            
            if card and hasattr(card, 'border_title'):
                icon = "↑" if new_direction is True else "↓" if new_direction is False else "○"
                # Preserve any fixed parameters info in the title
                current_title = card.border_title or ""
                if "(fixed:" in current_title:
                    # Extract fixed params info
                    fixed_part = current_title.split("(fixed:")[1].split(")")[0] if "(fixed:" in current_title else ""
                    card.border_title = f"{icon} Acquisition Function: {obj_name} (fixed: {fixed_part})" if fixed_part else f"{icon} Acquisition Function: {obj_name}"
                else:
                    card.border_title = f"{icon} Acquisition Function: {obj_name}"
        except Exception as e:
            logger.debug(f"Could not update direction button or border title: {e}")
        
        direction_text = "Maximize" if new_direction is True else "Minimize" if new_direction is False else "Undefined"
        self.notify(f"{obj_name}: {direction_text}", severity="information")
    
    @work(exclusive=True, thread=True)
    def update_objective_function_plots(self):
        """Update the objective function plots."""
        # Prevent duplicate execution - if already updating, skip
        if hasattr(self, '_updating_objective_plots') and self._updating_objective_plots:
            logger.debug("update_objective_function_plots already in progress, skipping duplicate call")
            return
        
        try:
            self._updating_objective_plots = True
            import numpy as np
            from tui.optimization.gp_regression import fit_gp_for_objective
            from tui.optimization.acquisition import compute_acquisition_function
            from tui.optimization.gp_multidim import fit_multidim_gp_for_objective, project_gp_predictions, get_active_parameters
            
            # Get filtered parameter sets with results
            filtered_sets = self.project.get_filtered_parameter_sets()
            sets_with_results = [ps for ps in filtered_sets if ps.result is not None]
            
            if not sets_with_results:
                def show_empty():
                    scroll_container = self.query_one("#plot-scroll-container", ScrollableContainer)
                    scroll_container.remove_children()  # Clear first
                    scroll_container.mount(Static("No results available. Calculate stats first."))
                self.app.call_from_thread(show_empty)
                self._updating_objective_plots = False  # Clear flag before returning
                return
            
            # Get all objective fields and remove duplicates
            objective_fields = list(dict.fromkeys(OptimizationResult.get_objective_fields()))  # Preserve order, remove duplicates
            logger.debug(f"Objective fields to plot: {objective_fields} (count: {len(objective_fields)})")
            
            # Get active parameters
            active_params = get_active_parameters(self.project)
            use_multidim = len(active_params) > 1
            
            # Primary parameter for visualization (usually diameter)
            primary_param = "diameter"
            if primary_param not in active_params and active_params:
                primary_param = active_params[0]
            
            # Collect all plot data first (in worker thread)
            plots_to_mount = []  # List of plot_data dictionaries
            
            # SIMPLIFIED: Create a plot for each objective - no duplicate tracking needed, just iterate
            for obj_name in objective_fields:
                # Collect data for this objective
                data = []
                for ps in sets_with_results:
                    # Check if all active params are present
                    if all(pname in ps.parameters for pname in active_params):
                        # CRITICAL: Verify we're getting the correct field value
                        obj_value = getattr(ps.result, obj_name, None)
                        if obj_value is None:
                            logger.warning(f"Objective '{obj_name}': getattr returned None for parameter set, using 0.0")
                            obj_value = 0.0
                        # Verify the value makes sense (not accidentally getting a different field)
                        if len(data) == 0:  # Log first value for verification
                            logger.debug(f"Objective '{obj_name}': first value from result: {obj_value} (type: {type(obj_value).__name__})")
                        data.append((ps.parameters, float(obj_value)))
                
                # Debug: Log data collection for ALL objectives to identify the issue
                sample_values = [val for _, val in data[:5]] if len(data) > 0 else []
                logger.info(f"Objective '{obj_name}': collected {len(data)} data points, sample values: {sample_values}, unique values: {len(set(val for _, val in data))}")
                
                if len(data) < 2:
                    logger.debug(f"Objective '{obj_name}': insufficient data points ({len(data)}), skipping")
                    continue  # Need at least 2 points
                
                # Verify we have unique data for this objective (not all zeros or same value)
                unique_values = set(val for _, val in data)
                if len(unique_values) == 1:
                    logger.debug(f"Objective '{obj_name}': all values are the same ({unique_values.pop()}), skipping")
                    continue  # Skip if all values are identical (no variation to plot)
                
                # Fit GP (multi-dimensional if multiple params active)
                obj_use_multidim = use_multidim and len(data) >= 3  # Need more points for multi-dim
                fixed_params = {}
                gp_model_stored = None  # Store GP model for suggestion predictions
                
                if obj_use_multidim:
                    gp_model = fit_multidim_gp_for_objective(
                        obj_name,
                        data,
                        active_params,
                        kernel_type=self.project.gp_kernel_type
                    )
                    
                    if gp_model is None:
                        # Fallback to 1D
                        obj_use_multidim = False
                    else:
                        gp_model_stored = gp_model  # Store for later use
                        # Project to 1D for visualization
                        # Use mean values of other parameters as fixed values
                        for pname in active_params:
                            if pname != primary_param:
                                values = [params[pname] for params, _ in data]
                                fixed_params[pname] = float(np.mean(values))
                        
                        gp_result = project_gp_predictions(
                            gp_model,
                            primary_param,
                            fixed_params=fixed_params if fixed_params else None,
                            num_points=50,
                            observed_data=data  # Pass observed data to determine range
                        )
                        
                        if gp_result is None:
                            obj_use_multidim = False
                        else:
                            X_pred, y_mean, y_std = gp_result
                            # Get observed values for this parameter
                            X_obs = np.array([params[primary_param] for params, _ in data])
                            y_obs = np.array([val for _, val in data])
                
                if not obj_use_multidim:
                    # Use 1D GP regression
                    gp_result = fit_gp_for_objective(
                        obj_name,
                        primary_param,
                        data,
                        kernel_type=self.project.gp_kernel_type
                    )
                    
                    if gp_result is None:
                        logger.debug(f"Objective '{obj_name}': GP fitting failed, skipping")
                        continue
                    
                    X_pred, y_mean, y_std, X_obs, y_obs = gp_result
                    # For 1D, we don't have a stored GP model, but we can use the predictions
                    gp_model_stored = None
                
                # Compute acquisition function
                best_value = max(y_obs) if len(y_obs) > 0 else 0.0
                acquisition = compute_acquisition_function(
                    y_mean,
                    y_std,
                    method=self.project.acquisition_method,
                    best_value=best_value
                )
                
                # Debug: Verify we have valid, unique data for this plot
                logger.debug(f"Objective '{obj_name}': created plot data with {len(X_pred)} prediction points, {len(X_obs)} observed points")
                
                # Create plot card
                obj_card = Container(classes="plot-card")
                
                # Get current direction
                current_direction = self.project.objective_directions.get(obj_name, None)
                if current_direction is None:
                    current_direction = OptimizationResult.get_objective_direction(obj_name)
                
                # Create direction button
                direction_button = Button(
                    id=f"direction-btn-{obj_name}",
                    classes="direction-btn"
                )
                
                if current_direction is True:
                    direction_button.label = "↑"
                    direction_button.tooltip = "Maximize (click to change)"
                elif current_direction is False:
                    direction_button.label = "↓"
                    direction_button.tooltip = "Minimize (click to change)"
                else:
                    direction_button.label = "○"
                    direction_button.tooltip = "Not used in Pareto (click to change)"
                
                # Set border title with fixed params info if multi-dim
                # Note: "fixed" parameters are used only for 1D visualization when the GP is multi-dimensional.
                # The GP model is fitted using all active parameters, but for display we project to 1D by
                # fixing other parameters at their mean values. This does NOT affect the GP fitting.
                direction_icon = "↑" if current_direction is True else "↓" if current_direction is False else "○"
                # CRITICAL: Use obj_name from current loop iteration, store it in the dictionary
                card_title = f"{direction_icon} Acquisition Function: {obj_name}"
                if obj_use_multidim and len(active_params) > 1 and fixed_params:
                    fixed_info = ", ".join([f"{p}={fixed_params.get(p, '?'):.2f}" for p in active_params if p != primary_param])
                    if fixed_info:
                        card_title += f" (fixed: {fixed_info})"
                obj_card.border_title = card_title
                logger.debug(f"Created plot card for objective '{obj_name}' with border title: {card_title}")
                
                # Store plot data for batch mounting (all at once on main thread)
                # SIMPLIFIED: Just store the data directly, no complex copying
                plots_to_mount.append({
                    'obj_name': obj_name,  # Simple string, no closure issues
                    'obj_card': obj_card,
                    'direction_button': direction_button,
                    'obj_use_multidim': obj_use_multidim,
                    'fixed_params': fixed_params.copy() if fixed_params else {},
                    'active_params': list(active_params) if isinstance(active_params, (list, tuple)) else active_params,
                    'primary_param': primary_param,
                    'X_pred': X_pred.copy() if hasattr(X_pred, 'copy') else X_pred,
                    'acquisition': acquisition.copy() if hasattr(acquisition, 'copy') else acquisition,
                    'y_mean': y_mean.copy() if hasattr(y_mean, 'copy') else y_mean,
                    'y_std': y_std.copy() if hasattr(y_std, 'copy') else y_std,
                    'X_obs': X_obs.copy() if hasattr(X_obs, 'copy') else X_obs,
                    'y_obs': y_obs.copy() if hasattr(y_obs, 'copy') else y_obs,
                    'gp_model': gp_model_stored,  # Store GP model for suggestion predictions
                })
            
            
            # Mount all plots at once on main thread (after clearing)
            def mount_all_plots():
                try:
                    scroll_container = self.query_one("#plot-scroll-container", ScrollableContainer)
                    
                    # Save scroll position
                    try:
                        self._saved_scroll_y = scroll_container.scroll_y if hasattr(scroll_container, 'scroll_y') else 0
                    except:
                        self._saved_scroll_y = 0
                    
                    # Clear existing plots FIRST
                    scroll_container.remove_children()
                    
                    # SIMPLIFIED: Just mount each plot once, no duplicate checking needed
                    logger.debug(f"Mounting {len(plots_to_mount)} plots")
                    
                    # Mount all plots
                    for idx, plot_data in enumerate(plots_to_mount):
                        plot_obj_name = plot_data['obj_name']  # Get directly from dict
                        obj_card = plot_data['obj_card']
                        direction_button = plot_data['direction_button']
                        
                        # Create plot widget (fresh instance for each plot)
                        plot_widget = PlotextStatic(classes="mini-plot")
                        
                        # Mount widgets
                        scroll_container.mount(obj_card)
                        obj_card.mount(direction_button)
                        obj_card.mount(plot_widget)
                        
                        # Convert all arrays to lists and validate lengths
                        # CRITICAL: Get arrays directly from plot_data to ensure we're using the correct data
                        x_pred_list = plot_data['X_pred'].tolist() if hasattr(plot_data['X_pred'], 'tolist') else list(plot_data['X_pred'])
                        acquisition_list = plot_data['acquisition'].tolist() if hasattr(plot_data['acquisition'], 'tolist') else list(plot_data['acquisition'])
                        y_mean_list = plot_data['y_mean'].tolist() if hasattr(plot_data['y_mean'], 'tolist') else list(plot_data['y_mean'])
                        x_obs_list = plot_data['X_obs'].tolist() if hasattr(plot_data['X_obs'], 'tolist') else list(plot_data['X_obs'])
                        y_obs_list = plot_data['y_obs'].tolist() if hasattr(plot_data['y_obs'], 'tolist') else list(plot_data['y_obs'])
                        
                        # Build plot title
                        plot_title = f"Acquisition Function A(x): {plot_obj_name}"
                        if plot_data['obj_use_multidim'] and len(plot_data['active_params']) > 1 and plot_data['fixed_params']:
                            fixed_info = ", ".join([f"{p}={plot_data['fixed_params'].get(p, '?'):.2f}" for p in plot_data['active_params'] if p != plot_data['primary_param']])
                            if fixed_info:
                                plot_title += f" (fixed: {fixed_info})"
                        
                        # CRITICAL FIX: Store plot configuration in widget so refresh_plot() can rebuild it
                        # This prevents all plots from sharing the same plotext state
                        plot_widget._plot_config = {
                            'title': plot_title,
                            'xlabel': f"Parameter: {plot_data['primary_param']}",
                            'ylabel': "Acquisition Value A(x)",
                            'x_pred': x_pred_list,
                            'acquisition': acquisition_list,
                            'y_mean': y_mean_list,
                            'x_obs': x_obs_list,
                            'y_obs': y_obs_list,
                            'y_std': plot_data['y_std'].tolist() if hasattr(plot_data['y_std'], 'tolist') else list(plot_data['y_std']) if plot_data['y_std'] is not None else None,
                        }
                        
                        # Now set up and render the plot
                        plt = plot_widget.plt
                        plt.clear_figure()
                        plt.theme("dark")
                        plt.title(plot_title)
                        plt.xlabel(f"Parameter: {plot_data['primary_param']}")
                        plt.ylabel("Acquisition Value A(x)")
                        
                        # Plot confidence intervals (μ ± σ) as subtle lines
                        try:
                            y_std = plot_data['y_std']
                            if (y_std is not None and 
                                len(y_std) > 0 and 
                                len(y_std) == len(y_mean_list) and
                                len(x_pred_list) == len(y_mean_list)):
                                y_std_list = y_std.tolist() if hasattr(y_std, 'tolist') else list(y_std)
                                y_upper = [(m + s) for m, s in zip(y_mean_list, y_std_list)]
                                y_lower = [(m - s) for m, s in zip(y_mean_list, y_std_list)]
                                
                                if len(x_pred_list) == len(y_upper) == len(y_lower):
                                    plt.plot(x_pred_list, y_upper, color=(38, 38, 38), label="")
                                    plt.plot(x_pred_list, y_lower, color=(38, 38, 38), label="")
                        except Exception as e:
                            logger.debug(f"Could not plot confidence intervals: {e}")
                        
                        # Plot only if arrays have matching lengths
                        if len(x_pred_list) == len(acquisition_list):
                            plt.plot(x_pred_list, acquisition_list, color="blue", label="A(x)")
                        if len(x_pred_list) == len(y_mean_list):
                            plt.plot(x_pred_list, y_mean_list, color="cyan", label="μ(x)")
                        if len(x_obs_list) == len(y_obs_list) and len(x_obs_list) > 0:
                            plt.scatter(x_obs_list, y_obs_list, color="red", label="Observed")
                        
                        # Overlay suggested points if available
                        current_suggestions = getattr(self.project, '_current_suggestions', [])
                        if current_suggestions:
                            try:
                                suggestion_x = []
                                suggestion_y = []
                                gp_model = plot_data.get('gp_model')
                                
                                for suggestion_params in current_suggestions:
                                    # Check if primary param is in suggestion
                                    if plot_data['primary_param'] in suggestion_params:
                                        x_sugg = suggestion_params[plot_data['primary_param']]
                                        
                                        # Predict using GP model if available
                                        if gp_model is not None and plot_data['obj_use_multidim']:
                                            # Multi-dimensional GP: construct full parameter vector
                                            param_array = []
                                            for pname in plot_data['active_params']:
                                                if pname in suggestion_params:
                                                    param_array.append(suggestion_params[pname])
                                                elif pname in plot_data['fixed_params']:
                                                    param_array.append(plot_data['fixed_params'][pname])
                                                else:
                                                    # Use default
                                                    if pname == "diameter":
                                                        param_array.append(30)
                                                    elif pname == "min_size":
                                                        param_array.append(15)
                                                    elif pname == "flow_threshold":
                                                        param_array.append(0.4)
                                                    elif pname == "cellprob_threshold":
                                                        param_array.append(0.0)
                                            
                                            if len(param_array) == len(plot_data['active_params']):
                                                try:
                                                    param_array_np = np.array(param_array).reshape(1, -1)
                                                    X_scaled = gp_model.scaler.transform(param_array_np)
                                                    y_pred, _ = gp_model.predict(X_scaled, return_std=True)
                                                    y_sugg = y_pred[0]
                                                except Exception as e:
                                                    logger.debug(f"Error predicting suggestion with GP: {e}")
                                                    # Fallback to closest point
                                                    closest_idx = min(range(len(x_pred_list)), 
                                                                     key=lambda i: abs(x_pred_list[i] - x_sugg))
                                                    y_sugg = y_mean_list[closest_idx]
                                            else:
                                                # Fallback
                                                closest_idx = min(range(len(x_pred_list)), 
                                                                 key=lambda i: abs(x_pred_list[i] - x_sugg))
                                                y_sugg = y_mean_list[closest_idx]
                                        else:
                                            # For 1D or no GP model, find closest prediction point
                                            closest_idx = min(range(len(x_pred_list)), 
                                                             key=lambda i: abs(x_pred_list[i] - x_sugg))
                                            y_sugg = y_mean_list[closest_idx]
                                        
                                        suggestion_x.append(x_sugg)
                                        suggestion_y.append(y_sugg)
                                
                                if suggestion_x and suggestion_y:
                                    # Plot suggestions with star marker and different color
                                    plt.scatter(suggestion_x, suggestion_y, marker="*", color="yellow", 
                                              label="Suggested", size=2)
                            except Exception as e:
                                logger.debug(f"Error plotting suggestions: {e}")
                        
                        plot_widget.refresh_plot()
                    
                    
                    # Restore scroll position after all plots are mounted
                    try:
                        if hasattr(self, '_saved_scroll_y') and hasattr(scroll_container, 'scroll_y') and self._saved_scroll_y > 0:
                            scroll_container.scroll_y = self._saved_scroll_y
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Error mounting plots: {e}", exc_info=True)
            
            # Mount all plots in a single main thread callback
            self.app.call_from_thread(mount_all_plots)
                
        except Exception as e:
            logger.error(f"Error updating objective function plots: {e}", exc_info=True)
        finally:
            # Clear the flag when done
            self._updating_objective_plots = False
    
    @work(exclusive=True, thread=True)
    def update_pareto_front_plots(self):
        """Update the Pareto front plots."""
        try:
            from tui.optimization.gp_regression import fit_gp_for_objective
            from tui.optimization.acquisition import compute_acquisition_function
            from tui.optimization.pareto import calculate_pareto_front
            from tui.optimization.pareto_metrics import calculate_hypervolume, find_knee_points
            
            # Clear plots on main thread (must be done on UI thread)
            def clear_plots():
                try:
                    scroll_container = self.query_one("#plot-scroll-container", ScrollableContainer)
                    # Save scroll position
                    try:
                        self._saved_scroll_y = scroll_container.scroll_y if hasattr(scroll_container, 'scroll_y') else 0
                    except:
                        self._saved_scroll_y = 0
                    # Clear existing plots
                    scroll_container.remove_children()
                except Exception as e:
                    logger.error(f"Error clearing plots: {e}")
                    self._saved_scroll_y = 0
            
            # Call clear on main thread
            self.app.call_from_thread(clear_plots)
            
            # Get filtered parameter sets with results
            filtered_sets = self.project.get_filtered_parameter_sets()
            sets_with_results = [ps for ps in filtered_sets if ps.result is not None]
            
            if not sets_with_results:
                def show_empty():
                    scroll_container = self.query_one("#plot-scroll-container", ScrollableContainer)
                    scroll_container.mount(Static("No results available. Calculate stats first."))
                self.app.call_from_thread(show_empty)
                return
            
            # Get objectives with defined directions
            objective_fields = OptimizationResult.get_objective_fields()
            enabled_objectives = []
            for obj_name in objective_fields:
                direction = self.project.objective_directions.get(obj_name, None)
                if direction is None:
                    direction = OptimizationResult.get_objective_direction(obj_name)
                if direction is not None:  # Only include if direction is defined
                    enabled_objectives.append(obj_name)
            
            if len(enabled_objectives) < 2:
                def show_message():
                    scroll_container = self.query_one("#plot-scroll-container", ScrollableContainer)
                    scroll_container.mount(Static("Need at least 2 objectives with defined directions. Use direction buttons in Objective Functions tab."))
                self.app.call_from_thread(show_message)
                return
            
            # Generate pairs of objectives
            objective_pairs = []
            for i in range(len(enabled_objectives)):
                for j in range(i + 1, len(enabled_objectives)):
                    objective_pairs.append((enabled_objectives[i], enabled_objectives[j]))
            
            # Primary parameter
            primary_param = "diameter"
            
            # Create a plot for each pair
            for obj1_name, obj2_name in objective_pairs:
                # Collect data for both objectives
                data1 = []
                data2 = []
                common_params = []
                
                for ps in sets_with_results:
                    if primary_param in ps.parameters:
                        obj1_value = getattr(ps.result, obj1_name, 0.0)
                        obj2_value = getattr(ps.result, obj2_name, 0.0)
                        data1.append((ps.parameters, obj1_value))
                        data2.append((ps.parameters, obj2_value))
                        common_params.append(ps.parameters)
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                
                # Fit GP for each objective
                gp1_result = fit_gp_for_objective(
                    obj1_name,
                    primary_param,
                    data1,
                    kernel_type=self.project.gp_kernel_type
                )
                gp2_result = fit_gp_for_objective(
                    obj2_name,
                    primary_param,
                    data2,
                    kernel_type=self.project.gp_kernel_type
                )
                
                if gp1_result is None or gp2_result is None:
                    continue
                
                # Build Pareto data points using acquisition function values or fallback to objective values
                pareto_data_points = []
                for i, params in enumerate(common_params):
                    param_value = params.get(primary_param, 0.0)
                    
                    # Try to get acquisition function values
                    try:
                        # Find corresponding GP predictions
                        X_pred1, y_mean1, y_std1, _, _ = gp1_result
                        X_pred2, y_mean2, y_std2, _, _ = gp2_result
                        
                        # Find closest prediction point
                        import numpy as np
                        idx1 = np.argmin(np.abs(X_pred1 - param_value))
                        idx2 = np.argmin(np.abs(X_pred2 - param_value))
                        
                        mean1, std1 = y_mean1[idx1], y_std1[idx1]
                        mean2, std2 = y_mean2[idx2], y_std2[idx2]
                        
                        best1 = max([v for _, v in data1])
                        best2 = max([v for _, v in data2])
                        
                        acq1 = compute_acquisition_function(
                            np.array([mean1]),
                            np.array([std1]),
                            method=self.project.acquisition_method,
                            best_value=best1
                        )[0]
                        
                        acq2 = compute_acquisition_function(
                            np.array([mean2]),
                            np.array([std2]),
                            method=self.project.acquisition_method,
                            best_value=best2
                        )[0]
                        
                        pareto_data_points.append((params, float(acq1), float(acq2)))
                    except Exception:
                        # Fallback to actual objective values
                        obj1_val = getattr(sets_with_results[i].result, obj1_name, 0.0)
                        obj2_val = getattr(sets_with_results[i].result, obj2_name, 0.0)
                        pareto_data_points.append((params, float(obj1_val), float(obj2_val)))
                
                if len(pareto_data_points) < 2:
                    continue
                
                # Get directions
                dir1 = self.project.objective_directions.get(obj1_name, None)
                if dir1 is None:
                    dir1 = OptimizationResult.get_objective_direction(obj1_name)
                dir2 = self.project.objective_directions.get(obj2_name, None)
                if dir2 is None:
                    dir2 = OptimizationResult.get_objective_direction(obj2_name)
                
                if dir1 is not None and dir2 is not None:
                    pareto_points, dominated_points = calculate_pareto_front(
                        obj1_name,
                        obj2_name,
                        pareto_data_points,
                        algorithm=self.project.pareto_algorithm,
                        maximize=[dir1, dir2]
                    )
                    
                    # Calculate hypervolume
                    hv_value = 0.0
                    if pareto_points:
                        hv_value = calculate_hypervolume(
                            pareto_points,
                            reference_point=None,
                            maximize=[dir1, dir2]
                        )
                    
                    # Find knee points
                    knee_indices = []
                    if pareto_points and len(pareto_points) >= 3:
                        knee_indices = find_knee_points(
                            pareto_points,
                            maximize=[dir1, dir2]
                        )
                    
                    # Create plot card with enhanced title
                    pareto_card = Container(classes="plot-card")
                    algo_name = self.project.pareto_algorithm.upper()
                    if algo_name == "NSGA2":
                        algo_name = "NSGA-II"
                    elif algo_name == "NSGA3":
                        algo_name = "NSGA-III"
                    title = f"Pareto Front: {obj1_name} vs {obj2_name} ({algo_name})"
                    if hv_value > 0:
                        title += f" | HV: {hv_value:.4f}"
                    pareto_card.border_title = title
                    
                    plot_widget = PlotextStatic(classes="mini-plot")
                    
                    # Capture variables for closure
                    plot_pareto_points = pareto_points
                    plot_dominated_points = dominated_points
                    plot_knee_indices = knee_indices
                    plot_hv_value = hv_value
                    
                    # Mount and update from main thread
                    def mount_and_update():
                        try:
                            scroll_container = self.query_one("#plot-scroll-container", ScrollableContainer)
                            scroll_container.mount(pareto_card)
                            
                            # Restore scroll position after mounting (if we saved it)
                            try:
                                if hasattr(self, '_saved_scroll_y') and hasattr(scroll_container, 'scroll_y') and self._saved_scroll_y > 0:
                                    scroll_container.scroll_y = self._saved_scroll_y
                            except:
                                pass  # Ignore if scroll position restoration fails
                            pareto_card.mount(plot_widget)
                            
                            # Update plot
                            plt = plot_widget.plt
                            plt.clear_figure()
                            plt.theme("dark")
                            algo_name = self.project.pareto_algorithm.upper()
                            if algo_name == "NSGA2":
                                algo_name = "NSGA-II"
                            elif algo_name == "NSGA3":
                                algo_name = "NSGA-III"
                            title = f"Pareto Front: {obj1_name} vs {obj2_name} ({algo_name})"
                            if plot_hv_value > 0:
                                title += f" | HV: {plot_hv_value:.4f}"
                            plt.title(title)
                            plt.xlabel(obj1_name)
                            plt.ylabel(obj2_name)
                            
                            if plot_dominated_points:
                                x_dom, y_dom = zip(*plot_dominated_points)
                                plt.scatter(list(x_dom), list(y_dom), color="gray", label="Dominated")
                            
                            if plot_pareto_points:
                                x_par, y_par = zip(*plot_pareto_points)
                                # Separate knee points from regular Pareto points
                                knee_x = []
                                knee_y = []
                                regular_x = []
                                regular_y = []
                                
                                for idx, (x, y) in enumerate(zip(x_par, y_par)):
                                    if idx in plot_knee_indices:
                                        knee_x.append(x)
                                        knee_y.append(y)
                                    else:
                                        regular_x.append(x)
                                        regular_y.append(y)
                                
                                if regular_x:
                                    plt.scatter(regular_x, regular_y, color="green", label="Pareto Front")
                                if knee_x:
                                    plt.scatter(knee_x, knee_y, color="yellow", label="Knee Points", marker="*")
                            
                            plot_widget.refresh_plot()
                        except Exception as e:
                            logger.error(f"Error mounting/plotting Pareto front: {e}", exc_info=True)
                    
                    self.app.call_from_thread(mount_and_update)
                    
        except Exception as e:
            logger.error(f"Error updating Pareto front plots: {e}", exc_info=True)
    
    @work(exclusive=True, thread=True)
    def show_suggestion_dialog(self, method: str = "acquisition"):
        """Show the parameter suggestion dialog.
        
        Args:
            method: Suggestion method ('acquisition', 'pareto', 'hybrid')
        """
        try:
            from tui.optimization.gp_suggestion import suggest_next_parameters_gp, compute_suggestion_rationale
            
            # Get filtered parameter sets with results for rationale computation
            filtered_sets = self.project.get_filtered_parameter_sets()
            sets_with_results = [ps for ps in filtered_sets if ps.result is not None]
            
            # Get suggestions
            suggestions = suggest_next_parameters_gp(
                self.project,
                method=method,
                num_suggestions=5
            )
            
            if not suggestions:
                self.notify("Could not generate suggestions. Need at least 5 scored parameter sets.", severity="warning")
                return
            
            # Generate rationales for each suggestion
            rationales = []
            for params in suggestions:
                rationale = compute_suggestion_rationale(
                    params,
                    self.project,
                    method,
                    sets_with_results=sets_with_results
                )
                rationales.append(rationale)
            
            # Store suggestions in project for visualization
            # We'll add a temporary field to track current suggestions
            if not hasattr(self.project, '_current_suggestions'):
                self.project._current_suggestions = []
            self.project._current_suggestions = suggestions.copy()
            
            # Show dialog
            def show_dialog():
                from tui.screens.suggestion_dialog import SuggestionDialog
                dialog = SuggestionDialog(suggestions, rationales)
                
                def handle_result(result):
                    if result:
                        # result is now a list of selected suggestions
                        if isinstance(result, list) and len(result) > 0:
                            # Batch config generation
                            if len(result) == 1:
                                # Single suggestion - use old method for compatibility
                                config_path = self.optimizer.create_iteration_config(result[0])
                                self.notify(f"Created config: {config_path}", severity="success")
                            else:
                                # Multiple suggestions - use batch method
                                config_path = self.optimizer.create_batch_iteration_config(result)
                                self.notify(f"Created batch config with {len(result)} parameter sets: {config_path}", severity="success")
                            
                            # Clear suggestions after config creation
                            if hasattr(self.project, '_current_suggestions'):
                                self.project._current_suggestions = []
                            
                            # Refresh to show new parameter sets
                            self.refresh_iterations()
                            
                            # Refresh objective function plots to remove suggestions
                            self.update_objective_function_plots()
                        else:
                            # Clear suggestions if cancelled
                            if hasattr(self.project, '_current_suggestions'):
                                self.project._current_suggestions = []
                            self.update_objective_function_plots()
                    else:
                        # Clear suggestions if cancelled
                        if hasattr(self.project, '_current_suggestions'):
                            self.project._current_suggestions = []
                        self.update_objective_function_plots()
                
                self.app.push_screen(dialog, handle_result)
            
            self.app.call_from_thread(show_dialog)
            
        except Exception as e:
            logger.error(f"Error showing suggestion dialog: {e}", exc_info=True)
            self.notify(f"Error generating suggestions: {e}", severity="error")


