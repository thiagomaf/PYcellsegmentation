"""Project dashboard screen for managing configs and optimization within a project."""
import logging
import re
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer, Grid
from textual.widgets import Button, Static, Label, Checkbox, SelectionList
from textual.widgets.selection_list import Selection
from textual.screen import Screen

from tui.optimization.models import OptimizationProject, ConfigFileInfo, ImagePoolEntry

logger = logging.getLogger(__name__)


def sanitize_widget_id(name: str) -> str:
    """
    Sanitize a string to be used as a widget ID.
    
    Widget IDs must contain only letters, numbers, underscores, or hyphens,
    and must not begin with a number.
    
    Args:
        name: The string to sanitize
        
    Returns:
        A sanitized string safe for use as a widget ID
    """
    # Replace all invalid characters (anything not alphanumeric, underscore, or hyphen) with hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '-', name)
    # Replace dots and spaces with hyphens (explicitly handle common cases)
    sanitized = sanitized.replace('.', '-').replace(' ', '-')
    # Consolidate multiple consecutive hyphens/underscores into a single hyphen
    sanitized = re.sub(r'[-_]+', '-', sanitized)
    # Remove leading/trailing hyphens and underscores
    sanitized = sanitized.strip('-_')
    # If the result starts with a number, prepend an underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    # If empty after sanitization, return a default value
    if not sanitized:
        sanitized = "item"
    return sanitized

class ProjectDashboard(Screen):
    """Dashboard for managing a loaded optimization project."""
    
    CSS_PATH = str(Path(__file__).parent / "project_dashboard.tcss")
    
    def on_screen_resume(self) -> None:
        """Called when screen is resumed (e.g., after returning from another screen)."""
        # Reload project from disk to get latest config files
        if self.project and self.project.filepath:
            try:
                from tui.optimization.models import OptimizationProject
                self.project = OptimizationProject.load(self.project.filepath)
            except Exception as e:
                logger.warning(f"Could not reload project: {e}")
        # Refresh lists to show any new config files
        # Use call_after_refresh to ensure screen is fully resumed before manipulating widgets
        self.call_after_refresh(self._populate_lists)
    
    def on_screen_suspend(self) -> None:
        """Called when screen is suspended (e.g., when pushing another screen)."""
    
    def __init__(self, project: OptimizationProject):
        super().__init__()
        self._mount_counter = 0  # Counter to ensure unique IDs across remounts
        try:
            super().__init__()
            self.project = project
        except Exception as e:
            raise
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the project dashboard."""
        project_name = Path(self.project.filepath).stem if self.project.filepath else "Unnamed Project"
        
        with Container(classes="project-dashboard-container"):
            yield Label(f"Project: {project_name}", classes="project-title")
            
            with ScrollableContainer(classes="project-dashboard-scroll"):
                with Horizontal(classes="sections_container"):
                    # Image Pool Section
                    with Vertical(classes="section"):
                        yield Label("Image Pool", classes="section-title")
                        yield Static("Manage images in the project pool.")
                        with Horizontal(classes="button-group"):
                            yield Button("Add Images",    id="add-images",    variant="primary")
                            yield Button("Remove Images", id="remove-images", variant="error")
                            yield Button("Select All",    id="select-all")
                            yield Button("Select None",   id="select-none")
                        
                        yield Label("Images in pool:", id="image-pool-label", classes="section-title")
                        yield Container(id="image-pool-list-container")
                    
                    # Config Management Section
                    with Vertical(classes="section"):
                        yield Label("Config Files", classes="section-title")
                        yield Static("Manage config files and include/exclude them from the pool.")
                        with Horizontal(classes="button-group"):
                            yield Button("New Config", id="new-config", variant="primary")
                            yield Button("Load Config", id="load-config", variant="default")
                            yield Button("Remove Config", id="remove-config", variant="error")
                        
                        yield Label("Config files:", classes="section-title")
                        yield Container(id="config-list-container")
                
                with Horizontal(classes="sections_container"):
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
    
    def on_unmount(self) -> None:
        """Called when screen is unmounted."""
    
    def on_unmount(self) -> None:
        """Called when screen is unmounted."""
    
    def _populate_lists(self) -> None:
        """Actually populate the lists - called after a delay to avoid duplicate ID issues."""
        try:
            logger.debug(f"Populating lists: image_pool={len(self.project.image_pool)}, config_files={len(self.project.config_files)}")
            
            # Update image pool label with counts
            try:
                active_count = sum(1 for entry in self.project.image_pool if entry.is_active)
                image_label = self.query_one("#image-pool-label", Label)
                image_label.update(f"Images in pool ({len(self.project.image_pool)} total, {active_count} active):")
            except Exception as e:
                logger.debug(f"Could not update image pool label: {e}")
            
            # Populate image pool
            try:
                image_list_container = self.query_one("#image-pool-list-container", Container)
                # Remove existing widget by querying for it and removing it explicitly
                # This ensures the widget ID is unregistered from Textual's global registry
                try:
                    existing_list = self.query_one("#image-pool-list", SelectionList)
                    existing_list.remove()
                except Exception:
                    # Widget doesn't exist, that's fine
                    pass
                # Also remove all children from container as backup
                image_list_container.remove_children()
                
                if self.project.image_pool:
                    logger.debug(f"Creating SelectionList with {len(self.project.image_pool)} image entries")
                    # Create Selection objects for each image entry
                    selections = []
                    for entry in self.project.image_pool:
                        image_name = Path(entry.filepath).name
                        # Use filepath as the value, and set initial state based on is_active
                        selections.append(Selection(image_name, entry.filepath, initial_state=entry.is_active))
                    
                    # Create new SelectionList with current selections
                    new_image_list = SelectionList(*selections, id="image-pool-list", classes="image-pool-list")
                    # Use call_after_refresh to ensure removal is processed before mounting
                    def mount_image_list():
                        try:
                            image_list_container.mount(new_image_list)
                        except Exception as e:
                            logger.error(f"Error mounting image list: {e}", exc_info=True)
                    self.call_after_refresh(mount_image_list)
                    # Track widget removal - add a callback to detect when it's removed
                    def on_image_list_removed():
                        try:
                            import json, time
                            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H6","location":"project_dashboard.py:_populate_lists","message":"Image list widget was removed","data":{},"timestamp":time.time()*1000})+"\n")
                        except: pass
                    # Store callback reference to prevent garbage collection
                    if not hasattr(self, '_widget_removal_callbacks'):
                        self._widget_removal_callbacks = []
                    self._widget_removal_callbacks.append((new_image_list, on_image_list_removed))
                    logger.debug(f"Mounted image SelectionList with {len(selections)} items")
                else:
                    logger.debug("No images in pool, creating empty SelectionList")
                    # Create empty SelectionList
                    new_image_list = SelectionList(id="image-pool-list", classes="image-pool-list")
                    image_list_container.mount(new_image_list)
            except Exception as e:
                logger.error(f"Error populating image pool list: {e}", exc_info=True)
            
            # Populate config list
            try:
                config_list_container = self.query_one("#config-list-container", Container)
                # Remove existing widget by querying for it and removing it explicitly
                # This ensures the widget ID is unregistered from Textual's global registry
                try:
                    existing_list = self.query_one("#config-list", SelectionList)
                    existing_list.remove()
                except Exception:
                    # Widget doesn't exist, that's fine
                    pass
                # Also remove all children from container as backup
                config_list_container.remove_children()
                
                if self.project.config_files:
                    logger.debug(f"Creating SelectionList with {len(self.project.config_files)} config entries")
                    # Create Selection objects for each config file
                    selections = []
                    for config_info in self.project.config_files:
                        config_name = Path(config_info.filepath).name
                        status = "✓" if config_info.included else "✗"
                        label_text = f"{status} {config_name}"
                        # Use filepath as the value, and set initial state based on included
                        selections.append(Selection(label_text, config_info.filepath, initial_state=config_info.included))
                    
                    # Create new SelectionList with current selections
                    new_config_list = SelectionList(*selections, id="config-list", classes="config-list")
                    # Use call_after_refresh to ensure removal is processed before mounting
                    def mount_config_list():
                        try:
                            config_list_container.mount(new_config_list)
                        except Exception as e:
                            logger.error(f"Error mounting config list: {e}", exc_info=True)
                    self.call_after_refresh(mount_config_list)
                    # Track widget removal - add a callback to detect when it's removed
                    def on_config_list_removed():
                        try:
                            import json, time
                            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H6","location":"project_dashboard.py:_populate_lists","message":"Config list widget was removed","data":{},"timestamp":time.time()*1000})+"\n")
                        except: pass
                    # Store callback reference to prevent garbage collection
                    if not hasattr(self, '_widget_removal_callbacks'):
                        self._widget_removal_callbacks = []
                    self._widget_removal_callbacks.append((new_config_list, on_config_list_removed))
                    logger.debug(f"Mounted config SelectionList with {len(selections)} items")
                else:
                    logger.debug("No config files in project, creating empty SelectionList")
                    # Create empty SelectionList
                    new_config_list = SelectionList(id="config-list", classes="config-list")
                    config_list_container.mount(new_config_list)
            except Exception as e:
                logger.error(f"Error populating config list: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error in _populate_lists: {e}", exc_info=True)
    
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
        elif event.button.id == "select-all":
            self.action_select_all_images()
        elif event.button.id == "select-none":
            self.action_select_none_images()
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
        from tui.screens.project_editor import ProjectEditorScreen
        # Create a new empty config
        from tui.models import ProjectConfig
        new_config = ProjectConfig()
        self.app.push_screen(ProjectEditorScreen(new_config, None))
    
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
        if not self.project.image_pool:
            self.notify("No images in pool to remove", severity="warning")
            return
        
        from tui.screens.image_selector import ImageSelector
        
        def on_selected(filepath: str):
            if filepath:
                # Remove the image from the project
                self.project.image_pool = [
                    entry for entry in self.project.image_pool 
                    if entry.filepath != filepath
                ]
                self.project.save()
                self._populate_lists()  # Refresh the image list
                image_name = Path(filepath).name
                self.notify(f"Removed image: {image_name}", severity="success")
        
        selector = ImageSelector(
            self.project.image_pool,
            title="Select Image to Remove"
        )
        self.app.push_screen(selector, on_selected)
    
    def action_select_all_images(self) -> None:
        """Select all images in the pool."""
        try:
            selection_list = self.query_one("#image-pool-list", SelectionList)
            # Use built-in select_all() method - this will trigger on_selection_list_selected_changed
            # which will automatically update the project state and label
            selection_list.select_all()
            self.notify(f"Selected all {len(self.project.image_pool)} image(s)", severity="success")
        except Exception as e:
            logger.error(f"Error selecting all images: {e}", exc_info=True)
            self.notify("Error selecting all images", severity="error")
    
    def action_select_none_images(self) -> None:
        """Deselect all images in the pool."""
        try:
            selection_list = self.query_one("#image-pool-list", SelectionList)
            # Use built-in deselect_all() method - this will trigger on_selection_list_selected_changed
            # which will automatically update the project state and label
            selection_list.deselect_all()
            self.notify(f"Deselected all {len(self.project.image_pool)} image(s)", severity="success")
        except Exception as e:
            logger.error(f"Error deselecting all images: {e}", exc_info=True)
            self.notify("Error deselecting all images", severity="error")
    
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
    
    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        """Handle selection changes in SelectionList widgets."""
        selection_list = event.selection_list
        selected_values = event.selection_list.selected
        
        # Determine which list was changed based on the widget ID
        if selection_list.id == "image-pool-list":
            # Update image pool entries based on selected values
            for entry in self.project.image_pool:
                entry.is_active = entry.filepath in selected_values
            
            self.project.save()
            # Update the label with new counts
            try:
                active_count = sum(1 for e in self.project.image_pool if e.is_active)
                image_label = self.query_one("#image-pool-label", Label)
                image_label.update(f"Images in pool ({len(self.project.image_pool)} total, {active_count} active):")
            except Exception:
                pass
        
        elif selection_list.id == "config-list":
            # Update config file entries based on selected values
            for config_info in self.project.config_files:
                config_info.included = config_info.filepath in selected_values
            
            self.project.save()
            # Refresh the list to update the status indicators
            self._populate_lists()
    
    def action_edit_parameters(self) -> None:
        """Edit parameter ranges."""
        from tui.screens.parameter_ranges_editor import ParameterRangesEditor
        
        def on_dismiss(saved: bool):
            if saved:
                # Save the project after parameter ranges are updated
                self.project.save()
                self.notify("Project saved with updated parameter ranges.", severity="success")
            
            # Reload project from disk to get latest config files
            if self.project and self.project.filepath:
                try:
                    from tui.optimization.models import OptimizationProject
                    self.project = OptimizationProject.load(self.project.filepath)
                except Exception as e:
                    logger.warning(f"Could not reload project: {e}")
            
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

