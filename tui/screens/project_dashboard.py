"""Project dashboard screen for managing configs and optimization within a project."""
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer, Grid
from textual.widgets import Button, Static, Label, Checkbox
from textual.screen import Screen

from tui.optimization.models import OptimizationProject, ConfigFileInfo, ImagePoolEntry

class ProjectDashboard(Screen):
    """Dashboard for managing a loaded optimization project."""
    
    CSS = """
    ProjectDashboard {
        layout: vertical;
    }
    
    .project-dashboard-container {
        width: 90%;
        height: 1fr;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
        layout: vertical;
    }
    
    .project-dashboard-scroll {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
    }
    
    .project-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 2;
        color: $primary;
    }
    
    .section {
        margin-top: 1;
        margin-bottom: 1;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
        color: $accent;
    }
    
    .button-group {
        margin-top: 1;
        height: auto;
        align: left middle;
    }
    
    .button-group Button {
        height: 3;
        margin: 0 1;
    }
    
    .config-list {
        height: auto;
        min-height: 3;
        max-height: 10;
        border: solid $secondary;
        padding: 1;
        margin-top: 1;
        layout: horizontal;
        overflow-y: auto;
    }
    
    .config-list Checkbox {
        width: auto;
        margin-right: 2;
    }
    
    .image-pool-list {
        height: auto;
        min-height: 3;
        max-height: 10;
        border: solid $secondary;
        padding: 1;
        margin-top: 1;
        overflow-y: auto;
    }
    
    .image-pool-list Checkbox {
        width: auto;
        margin: 0;
        margin-right: 2;
    }
    
    .section {
        height: auto;
    }
    
    Static {
        margin-bottom: 1;
    }
    """
    
    def __init__(self, project: OptimizationProject):
        super().__init__()
        self._mount_counter = 0  # Counter to ensure unique IDs across remounts
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"CSS2","location":"project_dashboard.py:88","message":"ProjectDashboard.__init__() entry - CSS will be parsed","data":{"css_length":len(self.CSS) if hasattr(self, 'CSS') else 0},"timestamp":__import__("time").time()*1000})+"\n")
        except: pass
        # #endregion
        try:
            super().__init__()
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"CSS2","location":"project_dashboard.py:92","message":"ProjectDashboard.__init__() completed - CSS parsed","data":{},"timestamp":__import__("time").time()*1000})+"\n")
            except: pass
            # #endregion
            self.project = project
        except Exception as e:
            # #region agent log
            try:
                with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    import json, traceback
                    error_msg = str(e)
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"CSS2","location":"project_dashboard.py:97","message":"ProjectDashboard.__init__() exception - possible CSS error","data":{"error":error_msg,"error_type":type(e).__name__,"traceback":traceback.format_exc(),"is_css_error":"CSS" in error_msg or "css" in error_msg.lower()},"timestamp":__import__("time").time()*1000})+"\n")
            except: pass
            # #endregion
            raise
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the project dashboard."""
        project_name = Path(self.project.filepath).stem if self.project.filepath else "Unnamed Project"
        
        with Container(classes="project-dashboard-container"):
            yield Label(f"Project: {project_name}", classes="project-title")
            
            with ScrollableContainer(classes="project-dashboard-scroll"):
                # Image Pool Section
                with Vertical(classes="section"):
                    yield Label("Image Pool", classes="section-title")
                    yield Static("Manage images in the project pool.")
                    with Horizontal(classes="button-group"):
                        yield Button("Add Images", id="add-images", variant="primary")
                        yield Button("Remove Images", id="remove-images", variant="error")
                    
                    yield Label("Images in pool:", id="image-pool-label", classes="section-title")
                    yield Container(classes="image-pool-list", id="image-pool-list")
                
                # Config Management Section
                with Vertical(classes="section"):
                    yield Label("Config Files", classes="section-title")
                    yield Static("Manage config files and include/exclude them from the pool.")
                    with Horizontal(classes="button-group"):
                        yield Button("New Config", id="new-config", variant="primary")
                        yield Button("Load Config", id="load-config", variant="default")
                        yield Button("Remove Config", id="remove-config", variant="error")
                    
                    yield Label("Config files:", classes="section-title")
                    yield Container(classes="config-list", id="config-list")
                
                # Parameter Ranges Section
                with Vertical(classes="section"):
                    yield Label("Parameter Ranges", classes="section-title")
                    yield Static("Configure the search space for optimization parameters.")
                    with Horizontal(classes="button-group"):
                        yield Button("Edit Parameter Ranges", id="edit-parameters", variant="primary")
                
                # Optimization Section
                with Vertical(classes="section"):
                    yield Label("Optimization", classes="section-title")
                    yield Static("Run and monitor parameter optimization.")
                    with Horizontal(classes="button-group"):
                        yield Button("View Optimization Dashboard", id="view-optimization", variant="primary")
                
                # Actions
                with Horizontal(classes="button-group"):
                    yield Button("Back to Main", id="back-main", variant="default")
                    yield Button("Save Project", id="save-project", variant="default")
    
    def on_mount(self) -> None:
        """Populate image pool and config lists."""
        # Increment counter to ensure unique IDs even if old widgets still exist
        self._mount_counter += 1
        # Use a small timer delay to ensure any pending widget removals complete first
        # This prevents duplicate ID errors when returning from other screens
        self.set_timer(0.1, self._populate_lists)
    
    def _populate_lists(self) -> None:
        """Actually populate the lists - called after a delay to avoid duplicate ID issues."""
        # Update image pool label with counts
        try:
            active_count = sum(1 for entry in self.project.image_pool if entry.is_active)
            image_label = self.query_one("#image-pool-label", Label)
            image_label.update(f"Images in pool ({len(self.project.image_pool)} total, {active_count} active):")
        except Exception:
            pass  # Label might not exist yet
        
        # Populate image pool
        image_list = self.query_one("#image-pool-list", Container)
        # First, query and remove any existing checkboxes with matching IDs to prevent duplicates
        existing_checkboxes = image_list.query(Checkbox)
        for checkbox in existing_checkboxes:
            try:
                checkbox.remove()
            except Exception:
                pass
        # Then remove all children
        try:
            image_list.remove_children()
        except Exception:
            pass
        
        if self.project.image_pool:
            for idx, entry in enumerate(self.project.image_pool):
                image_name = Path(entry.filepath).name
                # Create a unique ID by including index to avoid duplicates
                # This prevents issues when refreshing after returning from other screens
                # Include mount counter in ID to ensure uniqueness across remounts
                safe_id = f"img-{self._mount_counter}-{idx}-{image_name.replace('.', '-').replace(' ', '-')}"
                # Use checkbox to show active/inactive state
                checkbox = Checkbox(
                    image_name,
                    value=entry.is_active,
                    id=safe_id
                )
                # Store the filepath in a data attribute for retrieval
                checkbox.data = {"filepath": entry.filepath}
                image_list.mount(checkbox)
        else:
            image_list.mount(Static("No images in pool. Use 'Add Images' to add images."))
        
        # Populate config list
        config_list = self.query_one("#config-list", Container)
        # First, query and remove any existing checkboxes with matching IDs to prevent duplicates
        existing_checkboxes = config_list.query(Checkbox)
        for checkbox in existing_checkboxes:
            try:
                checkbox.remove()
            except Exception:
                pass
        # Then remove all children
        try:
            config_list.remove_children()
        except Exception:
            pass
        
        if self.project.config_files:
            for idx, config_info in enumerate(self.project.config_files):
                config_name = Path(config_info.filepath).name
                # Create a unique ID by including index to avoid duplicates
                # This prevents issues when refreshing after returning from other screens
                # Include mount counter in ID to ensure uniqueness across remounts
                safe_id = f"config-{self._mount_counter}-{idx}-{config_name.replace('.', '-').replace(' ', '-')}"
                status = "✓" if config_info.included else "✗"
                cb = Checkbox(f"{status} {config_name}", value=config_info.included, id=safe_id)
                config_list.mount(cb)
        else:
            config_list.mount(Static("No configs added yet. Use 'New Config' or 'Load Config' to add configs."))
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-main":
            self.app.pop_screen()
        elif event.button.id == "save-project":
            try:
                self.project.save()
                self.notify("Project saved successfully.", severity="success")
            except Exception as e:
                self.notify(f"Error saving project: {e}", severity="error")
        elif event.button.id == "add-images":
            self.action_add_images()
        elif event.button.id == "remove-images":
            self.action_remove_images()
        elif event.button.id == "new-config":
            self.action_new_config()
        elif event.button.id == "load-config":
            self.action_load_config()
        elif event.button.id == "remove-config":
            self.action_remove_config()
        elif event.button.id == "edit-parameters":
            self.action_edit_parameters()
        elif event.button.id == "view-optimization":
            self.action_view_optimization()
    
    def action_new_config(self) -> None:
        """Create a new config file."""
        from tui.screens.project_editor import ProjectEditor
        # Create a new empty config
        from tui.models import ProjectConfig
        new_config = ProjectConfig()
        self.app.push_screen(ProjectEditor(new_config, None))
    
    def action_add_images(self) -> None:
        """Add images to the image pool."""
        from tui.screens.image_picker import ImagePicker
        from pathlib import Path
        
        def on_selected(image_paths):
            if image_paths:
                images_added = 0
                existing_paths = {entry.filepath for entry in self.project.image_pool}
                
                for image_path in image_paths:
                    if image_path not in existing_paths:
                        entry = ImagePoolEntry(filepath=image_path, is_active=True)
                        self.project.image_pool.append(entry)
                        images_added += 1
                
                if images_added > 0:
                    self.project.save()
                    self.on_mount()  # Refresh
                    self.notify(f"{images_added} image(s) added to pool", severity="success")
                else:
                    self.notify("All selected images are already in the pool.", severity="warning")
        
        # Default to project root
        project_root = Path(__file__).parent.parent.parent
        self.app.push_screen(ImagePicker(initial_path=str(project_root)), on_selected)
    
    def action_remove_images(self) -> None:
        """Remove images from the pool."""
        # TODO: Show selection dialog
        self.notify("Remove images functionality - to be implemented", severity="info")
    
    def action_load_config(self) -> None:
        """Load a config file and add it to the project."""
        from tui.screens.file_picker import FilePicker
        from tui.models import ProjectConfig
        
        def on_selected(filepath):
            if filepath:
                # Check if config already exists
                existing = any(cf.filepath == filepath for cf in self.project.config_files)
                if not existing:
                    try:
                        # Load the config to extract images
                        config = ProjectConfig.from_json_file(filepath)
                        
                        # Extract image paths from the config
                        images_added = 0
                        existing_paths = {entry.filepath for entry in self.project.image_pool}
                        
                        for img_config in config.image_configurations:
                            image_path = img_config.original_image_filename
                            
                            # Resolve relative paths if needed
                            if not Path(image_path).is_absolute():
                                # Try to resolve relative to config file location first
                                config_dir = Path(filepath).parent
                                resolved_path = (config_dir / image_path).resolve()
                                
                                # If that doesn't exist, try relative to PROJECT_ROOT
                                if not resolved_path.exists():
                                    try:
                                        from src.file_paths import PROJECT_ROOT
                                        resolved_path = (Path(PROJECT_ROOT) / image_path).resolve()
                                    except (ImportError, Exception):
                                        pass
                                
                                if resolved_path.exists():
                                    image_path = str(resolved_path)
                            
                            # Add to image pool if not already present
                            if image_path not in existing_paths:
                                entry = ImagePoolEntry(filepath=image_path, is_active=img_config.is_active if hasattr(img_config, 'is_active') else True)
                                self.project.image_pool.append(entry)
                                images_added += 1
                        
                        # Add config to project (created_at will be set automatically by ConfigFileInfo)
                        config_info = ConfigFileInfo(filepath=filepath, included=True)
                        self.project.config_files.append(config_info)
                        self.project.save()
                        self.on_mount()  # Refresh the config list
                        
                        if images_added > 0:
                            self.notify(
                                f"Config added: {Path(filepath).name} ({images_added} images added to pool)",
                                severity="success"
                            )
                        else:
                            self.notify(f"Config added: {Path(filepath).name}", severity="success")
                    except Exception as e:
                        self.notify(f"Error loading config: {e}", severity="error")
                else:
                    self.notify("Config already in project.", severity="warning")
        
        self.app.push_screen(
            FilePicker(allowed_extensions={'.json'}, title="Select Configuration File"),
            on_selected
        )
    
    def action_remove_config(self) -> None:
        """Remove a config from the project."""
        if not self.project.config_files:
            self.notify("No config files to remove", severity="warning")
            return
        
        from tui.screens.config_selector import ConfigSelector
        
        def on_selected(filepath: str):
            if filepath:
                # Remove the config from the project
                self.project.config_files = [
                    cf for cf in self.project.config_files 
                    if cf.filepath != filepath
                ]
                self.project.save()
                self.on_mount()  # Refresh the config list
                config_name = Path(filepath).name
                self.notify(f"Removed config: {config_name}", severity="success")
        
        selector = ConfigSelector(
            self.project.config_files,
            title="Select Config File to Remove"
        )
        self.app.push_screen(selector, on_selected)
    
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes for configs and images."""
        checkbox = event.checkbox
        
        # Check if it's an image checkbox (has filepath in data)
        if hasattr(checkbox, 'data') and checkbox.data and 'filepath' in checkbox.data:
            # This is an image checkbox - use filepath from data
            filepath = checkbox.data['filepath']
            for entry in self.project.image_pool:
                if entry.filepath == filepath:
                    entry.is_active = checkbox.value
                    self.project.save()
                    # Update the label with new counts
                    try:
                        active_count = sum(1 for e in self.project.image_pool if e.is_active)
                        image_label = self.query_one("#image-pool-label", Label)
                        image_label.update(f"Images in pool ({len(self.project.image_pool)} total, {active_count} active):")
                    except Exception:
                        pass
                    break
        # Also handle by ID format: img-{mount}-{idx}-{name}
        elif checkbox.id and checkbox.id.startswith("img-"):
            # Parse the ID to get the index
            parts = checkbox.id.split("-")
            if len(parts) >= 3:
                try:
                    # Skip mount counter and get index (parts[2])
                    idx = int(parts[2])
                    if 0 <= idx < len(self.project.image_pool):
                        entry = self.project.image_pool[idx]
                        entry.is_active = checkbox.value
                        self.project.save()
                        # Update the label with new counts
                        try:
                            active_count = sum(1 for e in self.project.image_pool if e.is_active)
                            image_label = self.query_one("#image-pool-label", Label)
                            image_label.update(f"Images in pool ({len(self.project.image_pool)} total, {active_count} active):")
                        except Exception:
                            pass
                except (ValueError, IndexError):
                    pass
        elif checkbox.id and checkbox.id.startswith("config-"):
            # This is a config checkbox
            # The ID format is: config-{mount_counter}-{index}-{filename}
            # Parse the ID to get the index
            parts = checkbox.id.split("-")
            if len(parts) >= 3:
                try:
                    # Skip mount counter and get index (parts[2])
                    idx = int(parts[2])
                    if 0 <= idx < len(self.project.config_files):
                        config_info = self.project.config_files[idx]
                        config_name = Path(config_info.filepath).name
                        config_info.included = checkbox.value
                        self.project.save()
                        # Update the checkbox label to show status
                        status = "✓" if config_info.included else "✗"
                        checkbox.label = f"{status} {config_name}"
                except (ValueError, IndexError):
                    pass
    
    def action_edit_parameters(self) -> None:
        """Edit parameter ranges."""
        from tui.screens.parameter_ranges_editor import ParameterRangesEditor
        
        def on_dismiss(saved: bool):
            if saved:
                # Save the project after parameter ranges are updated
                self.project.save()
                self.notify("Project saved with updated parameter ranges.", severity="success")
            
            # Always refresh the dashboard to show any new config files that were generated
            # (configs are saved immediately when generated, so we need to refresh regardless)
            self.on_mount()
        
        # Pass the ranges object and project - edits will modify it in place
        editor = ParameterRangesEditor(self.project.parameter_ranges, project=self.project)
        self.app.push_screen(editor, on_dismiss)
    
    def action_view_optimization(self) -> None:
        """View the optimization dashboard."""
        from tui.screens.optimization_dashboard import OptimizationDashboard
        self.app.push_screen(OptimizationDashboard(self.project))

