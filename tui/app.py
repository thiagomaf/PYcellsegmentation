"""Main TUI application entry point."""
import sys
import argparse
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
    
    # Use CSS_PATH instead of pre-loading to avoid module-level file I/O on network drives
    CSS_PATH = "styles.tcss"
    
    TITLE = "PyCellSegmentation TUI"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("n", "new_project", "New Project"),
        ("l", "load_project", "Load Project"),
        ("o", "open_optimization", "Optimization"),
        ("f12",    "inspector", "Inspector"),  # Open widget inspector (F12 or Ctrl+I)
        ("ctrl+i", "inspector", "Inspector"),  # Alternative key binding
        ("escape", "back", "Back"),  # For project editor
        ("ctrl+s", "save", "Save"),  # For project editor
        ("1", "view_general", "General"),  # For project editor
        ("2", "view_images", "Images"),  # For project editor
        ("3", "view_parameters", "Parameters"),  # For project editor
        ("4", "view_preview", "Preview"),  # For project editor
    ]
    
    def __init__(self, project_filepath: Optional[str] = None, initial_view: Optional[str] = None):
        """Initialize the application.
        
        Args:
            project_filepath: Optional path to a project file to load on startup
            initial_view: Optional view to open after loading project (e.g., 'parameters', 'optimization')
        """
        import time
        import sys
        
        # Check if numpy is imported at module level (should NOT be) - moved to runtime
        # #region agent log
        try:
            numpy_imported = 'numpy' in sys.modules
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF6","location":"app.py:__init__","message":"Module-level import check","data":{"numpy_imported":numpy_imported},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        
        init_start = time.time()
        try:
            super_start = time.time()
            super().__init__()
            super_done = time.time()
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF3","location":"app.py:__init__","message":"super().__init__() completed","data":{"elapsed_ms":(super_done-super_start)*1000},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            self.current_project: ProjectConfig | None = None
            self.current_filepath: str | None = None
            self._content_area: Container | None = None
            self.initial_project_filepath = project_filepath
            self.initial_view = initial_view
            init_done = time.time()
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF3","location":"app.py:__init__","message":"PyCellSegTUI.__init__() completed","data":{"total_elapsed_ms":(init_done-init_start)*1000},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
        except Exception as e:
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json, traceback
                    error_msg = str(e)
                    error_type = type(e).__name__
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF3","location":"app.py:95","message":"PyCellSegTUI.__init__() exception","data":{"error":error_msg,"error_type":error_type,"traceback":traceback.format_exc()},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            raise
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(id="app-header")
        yield Container(id="content-area")
        yield Footer(id="app-footer")
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        import time
        mount_start = time.time()
        
        # If a project filepath was provided via command line, load it
        if hasattr(self, 'initial_project_filepath') and self.initial_project_filepath:
            try:
                load_start = time.time()
                from tui.optimization.models import OptimizationProject
                project = OptimizationProject.load(self.initial_project_filepath)
                load_done = time.time()
                # #region agent log
                try:
                    with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        import json
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF3","location":"app.py:140","message":"project loaded","data":{"elapsed_ms":(load_done-load_start)*1000},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion
                
                # Check if a specific view was requested
                if hasattr(self, 'initial_view') and self.initial_view:
                    self._open_initial_view(project, self.initial_view)
                else:
                    self.show_project_dashboard(project)
                return
            except Exception as e:
                self.notify(f"Error loading project from command line: {e}", severity="error")
                # Fall through to show dashboard
        
        try:
            # Get reference to content area
            query_start = time.time()
            self._content_area = self.query_one("#content-area", Container)
            query_done = time.time()
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF3","location":"app.py:162","message":"content area queried","data":{"elapsed_ms":(query_done-query_start)*1000},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            # Mount the dashboard in the content area
            dashboard_start = time.time()
            self.show_dashboard()
            dashboard_done = time.time()
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF3","location":"app.py:172","message":"show_dashboard() completed","data":{"elapsed_ms":(dashboard_done-dashboard_start)*1000,"total_mount_ms":(dashboard_done-mount_start)*1000},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
        except Exception as e:
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json, traceback
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF3","location":"app.py:181","message":"on_mount() exception","data":{"error":str(e),"traceback":traceback.format_exc()},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            raise
    
    def show_dashboard(self) -> None:
        """Show the dashboard in the content area."""
        import time
        if self._content_area is None:
            self._content_area = self.query_one("#content-area", Container)
        
        # Clear content area and mount dashboard
        clear_start = time.time()
        self._content_area.remove_children()
        clear_done = time.time()
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF3","location":"app.py:191","message":"children removed","data":{"elapsed_ms":(clear_done-clear_start)*1000},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        
        create_start = time.time()
        dashboard = Dashboard()
        create_done = time.time()
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF3","location":"app.py:199","message":"Dashboard created","data":{"elapsed_ms":(create_done-create_start)*1000},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        
        mount_start = time.time()
        self._content_area.mount(dashboard)
        mount_done = time.time()
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF3","location":"app.py:207","message":"Dashboard mounted","data":{"elapsed_ms":(mount_done-mount_start)*1000},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        
        # Update header and footer to dashboard state
        self.update_header("Project Manager", "")
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
        config_name = "New Config" if filepath is None else Path(filepath).name
        self.update_header("PyCellSegmentation TUI", f"Editing: {config_name}")
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
            self.action_new_optimization_project()
        elif button_id == "load-project":
            self.action_load_optimization_project()
        elif button_id == "exit":
            self.action_quit()
    
    def action_new_config(self) -> None:
        """Create a new config."""
        self.show_project_editor()
    
    def action_load_config(self) -> None:
        """Load an existing config."""
        from tui.screens.file_picker import FilePicker
        
        def on_dismiss(filepath):
            if filepath:
                self.load_config(filepath)
        
        self.push_screen(FilePicker(), on_dismiss)
        
    def action_open_optimization(self) -> None:
        """Open the optimization setup screen."""
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"app.py:186","message":"action_open_optimization() entry","data":{},"timestamp":__import__("time").time()*1000})+"\n")
        except: pass
        # #endregion
        try:
            from tui.screens.optimization_setup import OptimizationSetup
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"app.py:190","message":"OptimizationSetup imported","data":{},"timestamp":__import__("time").time()*1000})+"\n")
            except: pass
            # #endregion
            screen = OptimizationSetup()
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"app.py:194","message":"OptimizationSetup() created","data":{},"timestamp":__import__("time").time()*1000})+"\n")
            except: pass
            # #endregion
            self.push_screen(screen)
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"app.py:198","message":"push_screen() completed","data":{},"timestamp":__import__("time").time()*1000})+"\n")
            except: pass
            # #endregion
        except Exception as e:
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json, traceback
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"app.py:201","message":"action_open_optimization() exception","data":{"error":str(e),"traceback":traceback.format_exc()},"timestamp":__import__("time").time()*1000})+"\n")
            except: pass
            # #endregion
            raise
    
    def action_new_optimization_project(self) -> None:
        """Create a new project."""
        from tui.screens.optimization_setup import OptimizationSetup
        self.push_screen(OptimizationSetup())
    
    def action_load_optimization_project(self) -> None:
        """Load an existing project."""
        from tui.screens.file_picker import FilePicker
        from tui.optimization.models import OptimizationProject
        from pathlib import Path
        
        def on_dismiss(filepath):
            if filepath and filepath.endswith(".opt.json"):
                try:
                    project = OptimizationProject.load(filepath)
                    self.show_project_dashboard(project)
                except Exception as e:
                    self.notify(f"Error loading project: {e}", severity="error")
            elif filepath:
                self.notify("Please select a .opt.json file", severity="error")
        
        # Default to project root instead of config directory
        project_root = Path(__file__).parent.parent
        self.push_screen(FilePicker(initial_path=str(project_root)), on_dismiss)
    
    def show_project_dashboard(self, project) -> None:
        """Show the project dashboard for a loaded project."""
        from tui.screens.project_dashboard import ProjectDashboard
        self.push_screen(ProjectDashboard(project))
    
    def _open_initial_view(self, project, view_name: str) -> None:
        """Open a specific view after loading a project.
        
        Args:
            project: The loaded OptimizationProject
            view_name: Name of the view to open (e.g., 'parameters', 'optimization')
        """
        view_name_lower = view_name.lower()
        
        if view_name_lower in ['parameters', 'params', 'param']:
            # Open parameter ranges editor
            from tui.screens.parameter_ranges_editor import ParameterRangesEditor
            from tui.screens.project_dashboard import ProjectDashboard
            
            # First show the project dashboard, then open the parameter editor
            dashboard = ProjectDashboard(project)
            self.push_screen(dashboard)
            
            # Open parameter editor after dashboard is mounted
            def open_editor():
                editor = ParameterRangesEditor(project.parameter_ranges, project=project)
                self.push_screen(editor)
            
            # Use call_after_refresh to ensure dashboard is mounted first
            self.call_after_refresh(open_editor)
            
        elif view_name_lower in ['optimization', 'opt', 'optimize']:
            # Open optimization dashboard
            from tui.screens.optimization_dashboard import OptimizationDashboard
            self.push_screen(OptimizationDashboard(project))
            
        else:
            # Unknown view, just show project dashboard
            self.notify(f"Unknown view: {view_name}. Showing project dashboard instead.", severity="warning")
            self.show_project_dashboard(project)
    
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
    
    def load_config(self, filepath: str) -> None:
        """Load a config from a file (for config editor, not optimization project)."""
        try:
            config = ProjectConfig.from_json_file(filepath)
            self.show_project_editor(config, filepath)
        except Exception as e:
            self.notify(f"Error loading config: {e}", severity="error")


def main():
    """Main entry point for the TUI."""
    parser = argparse.ArgumentParser(
        description="PyCellSegmentation TUI - Textual User Interface for managing segmentation projects"
    )
    parser.add_argument(
        "-p", "--project",
        type=str,
        help="Path to a project file (.opt.json) to load on startup"
    )
    parser.add_argument(
        "--view",
        type=str,
        help="View to open after loading project: 'parameters' (or 'params') to open parameter editor, 'optimization' (or 'opt') to open optimization dashboard"
    )
    # Also support --parameters as a convenience flag
    parser.add_argument(
        "--parameters",
        action="store_const",
        const="parameters",
        dest="view",
        help="Shortcut to open parameter editor (equivalent to --view parameters)"
    )
    # Also support --optimization as a convenience flag
    parser.add_argument(
        "--optimization",
        action="store_const",
        const="optimization",
        dest="view",
        help="Shortcut to open optimization dashboard (equivalent to --view optimization)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="tui_console.log",
        help="Path to log file for capturing console output (default: tui_console.log)"
    )
    
    args = parser.parse_args()
    
    # Set up file logging if log-file is specified
    if args.log_file:
        import logging
        import os
        from datetime import datetime
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else "."
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Set up file handler for all loggers
        file_handler = logging.FileHandler(args.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)
        
        logging.info(f"Logging to file: {args.log_file}")
    
    # #region agent log
    try:
        with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"app.py:298","message":"main() entry","data":{"args":vars(args)},"timestamp":__import__("time").time()*1000})+"\n")
    except: pass
    # #endregion
    
    # Validate project filepath if provided
    project_filepath = None
    if args.project:
        project_path = Path(args.project)
        if not project_path.exists():
            print(f"Error: Project file not found: {args.project}", file=sys.stderr)
            sys.exit(1)
        if not project_path.suffix == ".opt.json":
            print(f"Warning: Project file should have .opt.json extension: {args.project}", file=sys.stderr)
        project_filepath = str(project_path.resolve())
    
    try:
        import time
        app_creation_start = time.time()
        app = PyCellSegTUI(project_filepath=project_filepath, initial_view=args.view)
        app_creation_done = time.time()
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF4","location":"app.py:567","message":"PyCellSegTUI() created","data":{"elapsed_ms":(app_creation_done-app_creation_start)*1000,"has_project":project_filepath is not None},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        run_start = time.time()
        app.run()
        run_done = time.time()
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"PERF4","location":"app.py:575","message":"app.run() completed","data":{"elapsed_ms":(run_done-run_start)*1000},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
    except Exception as e:
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json, traceback
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"app.py:315","message":"main() exception","data":{"error":str(e),"traceback":traceback.format_exc()},"timestamp":__import__("time").time()*1000})+"\n")
        except: pass
        # #endregion
        raise


if __name__ == "__main__":
    main()
