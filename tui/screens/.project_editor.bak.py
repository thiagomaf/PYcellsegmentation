"""Project editor screen for editing configuration files."""
from pathlib import Path as PathLib
from typing import Optional
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static

from tui.models import ProjectConfig


class ProjectEditor(Container):
    """Main project editor container."""
    
    # CSS_PATH = str(PathLib(__file__).parent / "project_editor.tcss")
    CSS_PATH = str(PathLib(__file__).parent / "screens/project_editor.tcss")
    
    def __init__(self, config: Optional[ProjectConfig] = None, filepath: Optional[str] = None):
        """Initialize the project editor."""
        super().__init__()
        if config is None:
            self.config = ProjectConfig()
        else:
            self.config = config
        self.filepath = filepath
        self.current_view = "general"
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the project editor."""
        with Horizontal(id="editor-container"):
            with Vertical(id="editor-sidebar"):
                yield Static("Navigation", classes="nav-btn", id="nav-title")
                # Use Button with safe nav-btn class
                yield Button("▶ Config Editor",  id="nav-config",   classes="nav-btn nav-btn--selected")
                yield Button("  Pipeline Status", id="nav-pipeline", classes="nav-btn")
                yield Button("  Result Explorer", id="nav-results",  classes="nav-btn")


        ##########

        # with Horizontal(classes="editor-header"):
        #     # title = "New Project" if self.filepath is None else Path(self.filepath).name
        #     # yield Static(title, classes="editor-title")
        #     with Container(classes="editor-actions"):
        #         yield Button("Save", id="save-button", classes="toolbar-button", variant="primary")
        
        # # Main content area
        # with Horizontal():
        #     # Sidebar
        #     #with Vertical(classes="editor-sidebar"):
        #     with Vertical(classes="config-sidebar"):
        #         yield Static("Navigation", classes="nav-btn", id="nav-title")
        #         # Use Button with safe nav-btn class
        #         yield Button("▶ General",   id="nav-general", classes="nav-btn nav-btn--selected")
        #         yield Button("  Images",     id="nav-images", classes="nav-btn")
        #         yield Button("  Parameters", id="nav-parameters", classes="nav-btn")
        #         yield Button("  Preview",    id="nav-preview", classes="nav-btn")
            
        #     # Content area
        #     with Container(classes="editor-content", id="content-area"):
        #         yield Static("Loading...", id="content-view")
        
    def on_mount(self) -> None:
        """Called when the container is mounted."""
        # # Update app header and footer
        # self._update_header_footer()

        # self.show_view("config")

        # Map button IDs to view names
        button_to_view = {
            "nav-config": "config",
            "nav-pipeline": "pipeline", 
            "nav-results": "results"
        }
        
        # Get the selected button and show corresponding view
        try:
            selected_button = self.query_one(".nav-btn--selected", Button)
            button_id = selected_button.id
            view_name = button_to_view.get(button_id, "config")  # Default to "general" if not found
            self.show_view(view_name)
        except Exception:
            # If no button is selected or query fails, use default
            self.show_view("config")
        
        

        # # #region agent log
        # import json
        # log_path = PathLib(__file__).parent.parent.parent / ".cursor" / "debug.log"
        # try:
        #     with open(log_path, "a", encoding="utf-8") as f:
        #         f.write(json.dumps({"sessionId":"debug-session","runId":"button-fix","hypothesisId":"K","location":"tui/screens/project_editor.py:133","message":"on_mount called","data":{},"timestamp":int(__import__("time").time()*1000)}) + "\n")
        # except: pass
        # # #endregion
        
        # try:
        #     self.show_view("general")
        #     # #region agent log
        #     try:
        #         with open(log_path, "a", encoding="utf-8") as f:
        #             f.write(json.dumps({"sessionId":"debug-session","runId":"button-fix","hypothesisId":"K","location":"tui/screens/project_editor.py:142","message":"on_mount show_view completed","data":{},"timestamp":int(__import__("time").time()*1000)}) + "\n")
        #     except: pass
        #     # #endregion
        # except Exception as e:
        #     # #region agent log
        #     try:
        #         with open(log_path, "a", encoding="utf-8") as f:
        #             f.write(json.dumps({"sessionId":"debug-session","runId":"button-fix","hypothesisId":"K","location":"tui/screens/project_editor.py:145","message":"on_mount exception","data":{"exception_type":str(type(e).__name__),"exception_msg":str(e)},"timestamp":int(__import__("time").time()*1000)}) + "\n")
        #     except: pass
        #     # #endregion
        #     # Re-raise so we can see the error
        #     raise
    
    # def show_view(self, view_name: str) -> None:
    #     """Switch to a different view."""
    #     # #region agent log
    #     import json
    #     log_path = PathLib(__file__).parent.parent.parent / ".cursor" / "debug.log"
    #     try:
    #         with open(log_path, "a", encoding="utf-8") as f:
    #             f.write(json.dumps({"sessionId":"debug-session","runId":"button-fix","hypothesisId":"J","location":"tui/screens/project_editor.py:137","message":"show_view called","data":{"view_name":view_name,"current_view":self.current_view},"timestamp":int(__import__("time").time()*1000)}) + "\n")
    #     except: pass
    #     # #endregion
        
    #     self.current_view = view_name
        
    #     # Update sidebar selection
    #     labels = {
    #         "nav-general":    "▶ General",
    #         "nav-images":     "  Images",
    #         "nav-parameters": "  Parameters",
    #         "nav-preview":    "  Preview"
    #     }
        
    #     for nav_id in ["nav-general", "nav-images", "nav-parameters", "nav-preview"]:
    #         try:
    #             btn = self.query_one(f"#{nav_id}", Button)
    #         except Exception as e:
    #             # #region agent log
    #             try:
    #                 with open(log_path, "a", encoding="utf-8") as f:
    #                     f.write(json.dumps({"sessionId":"debug-session","runId":"button-fix","hypothesisId":"J","location":"tui/screens/project_editor.py:160","message":"failed to query button","data":{"nav_id":nav_id,"error":str(e)},"timestamp":int(__import__("time").time()*1000)}) + "\n")
    #             except: pass
    #             # #endregion
    #             # If button not found, skip it
    #             continue
            
    #         # #region agent log
    #         try:
    #             before_classes = set(btn.classes) if hasattr(btn, 'classes') else set()
    #             before_label = btn.label if hasattr(btn, 'label') else 'N/A'
    #             with open(log_path, "a", encoding="utf-8") as f:
    #                 f.write(json.dumps({"sessionId":"debug-session","runId":"button-fix","hypothesisId":"J","location":"tui/screens/project_editor.py:170","message":"before button update","data":{"nav_id":nav_id,"before_classes":list(before_classes),"before_label":before_label,"is_selected":nav_id == f"nav-{view_name}"},"timestamp":int(__import__("time").time()*1000)}) + "\n")
    #         except: pass
    #         # #endregion
            
    #         try:
    #             if nav_id == f"nav-{view_name}":
    #                 # This button should be selected
    #                 btn.add_class("nav-btn--selected")
    #                 # Update label to show selection - ensure it starts with ▶
    #                 try:
    #                     # Try to get current label - Button might use different property
    #                     if hasattr(btn, 'label'):
    #                         current_label = str(btn.label)
    #                     elif hasattr(btn, 'renderable'):
    #                         current_label = str(btn.renderable)
    #                     else:
    #                         # Fallback: use the original label from dict
    #                         current_label = labels[nav_id]
                        
    #                     # Strip any leading whitespace or ▶, then add ▶
    #                     text_part = current_label.lstrip("▶").lstrip()
                        
    #                     # Try to set label
    #                     if hasattr(btn, 'label'):
    #                         btn.label = f"▶ {text_part}"
    #                     elif hasattr(btn, 'update'):
    #                         btn.update(f"▶ {text_part}")
    #                 except Exception as label_error:
    #                     # #region agent log
    #                     try:
    #                         with open(log_path, "a", encoding="utf-8") as f:
    #                             f.write(json.dumps({"sessionId":"debug-session","runId":"button-fix","hypothesisId":"L","location":"tui/screens/project_editor.py:215","message":"label update exception","data":{"nav_id":nav_id,"error":str(label_error)},"timestamp":int(__import__("time").time()*1000)}) + "\n")
    #                     except: pass
    #                     # #endregion
    #                     # If label update fails, at least the class is set correctly
    #                     pass
    #             else:
    #                 # This button should not be selected
    #                 btn.remove_class("nav-btn--selected")
    #                 # Update label to remove selection indicator - ensure it starts with spaces
    #                 try:
    #                     # Try to get current label
    #                     if hasattr(btn, 'label'):
    #                         current_label = str(btn.label)
    #                     elif hasattr(btn, 'renderable'):
    #                         current_label = str(btn.renderable)
    #                     else:
    #                         # Fallback: use the original label from dict
    #                         current_label = labels[nav_id]
                        
    #                     # Strip any leading whitespace or ▶, then add spaces
    #                     text_part = current_label.lstrip("▶").lstrip()
                        
    #                     # Try to set label
    #                     if hasattr(btn, 'label'):
    #                         btn.label = f"  {text_part}"
    #                     elif hasattr(btn, 'update'):
    #                         btn.update(f"  {text_part}")
    #                 except Exception as label_error:
    #                     # #region agent log
    #                     try:
    #                         with open(log_path, "a", encoding="utf-8") as f:
    #                             f.write(json.dumps({"sessionId":"debug-session","runId":"button-fix","hypothesisId":"L","location":"tui/screens/project_editor.py:240","message":"label update exception","data":{"nav_id":nav_id,"error":str(label_error)},"timestamp":int(__import__("time").time()*1000)}) + "\n")
    #                     except: pass
    #                     # #endregion
    #                     # If label update fails, at least the class is set correctly
    #                     pass
                
    #             # #region agent log
    #             try:
    #                 after_classes = set(btn.classes) if hasattr(btn, 'classes') else set()
    #                 after_label = str(btn.label) if hasattr(btn, 'label') else (str(btn.renderable) if hasattr(btn, 'renderable') else 'N/A')
    #                 with open(log_path, "a", encoding="utf-8") as f:
    #                     f.write(json.dumps({"sessionId":"debug-session","runId":"button-fix","hypothesisId":"J","location":"tui/screens/project_editor.py:250","message":"after button update","data":{"nav_id":nav_id,"after_classes":list(after_classes),"after_label":after_label,"has_selected_class":"nav-btn--selected" in after_classes},"timestamp":int(__import__("time").time()*1000)}) + "\n")
    #             except: pass
    #             # #endregion
    #         except Exception as e:
    #             # #region agent log
    #             try:
    #                 with open(log_path, "a", encoding="utf-8") as f:
    #                     f.write(json.dumps({"sessionId":"debug-session","runId":"button-fix","hypothesisId":"J","location":"tui/screens/project_editor.py:255","message":"button update exception","data":{"nav_id":nav_id,"error":str(e),"error_type":str(type(e).__name__)},"timestamp":int(__import__("time").time()*1000)}) + "\n")
    #             except: pass
    #             # #endregion
    #             # Don't fail completely if one button fails - just log it
    #             # The screen should still work even if button labels can't be updated
    #             pass
        
    #     # Update content area
    #     content_area = self.query_one("#content-area", Container)
    #     content_area.remove_children()
        
    #     if view_name == "general":
    #         from tui.widgets.general_view import GeneralView
    #         content_area.mount(GeneralView(self.config))
    #     elif view_name == "images":
    #         from tui.widgets.images_view import ImagesView
    #         images_view = ImagesView(self.config)
    #         content_area.mount(images_view)
    #         # Store reference for refreshing
    #         self._images_view = images_view
    #     elif view_name == "parameters":
    #         from tui.widgets.parameters_view import ParametersView
    #         params_view = ParametersView(self.config)
    #         content_area.mount(params_view)
    #         # Store reference for refreshing
    #         self._parameters_view = params_view
    #     elif view_name == "preview":
    #         from tui.widgets.preview_view import PreviewView
    #         preview_view = PreviewView(self.config)
    #         content_area.mount(preview_view)
    #         # Store reference for refreshing
    #         self._preview_view = preview_view
    #         # Refresh preview when switching to it
    #         preview_view.refresh_preview()
    
    # def on_button_pressed(self, event: Button.Pressed) -> None:
    #     """Handle button presses."""
    #     button_id = event.button.id
        
    #     if button_id == "save-button":
    #         self.action_save()
    #     elif button_id and button_id.startswith("nav-"):
    #         view_name = button_id.replace("nav-", "")
    #         self.show_view(view_name)
    
    # def action_view_general(self) -> None:
    #     """Switch to general view."""
    #     self.show_view("general")
    
    # def action_view_images(self) -> None:
    #     """Switch to images view."""
    #     self.show_view("images")
    
    # def action_view_parameters(self) -> None:
    #     """Switch to parameters view."""
    #     self.show_view("parameters")
    
    # def action_view_preview(self) -> None:
    #     """Switch to preview view."""
    #     self.show_view("preview")
    
    # def action_save(self) -> None:
    #     """Save the current project."""
    #     if self.filepath is None:
    #         # Need to get filepath from user
    #         from tui.screens.save_dialog import SaveDialog
            
    #         def on_dismiss(filepath):
    #             if filepath:
    #                 self.filepath = filepath
    #                 self.app.current_filepath = filepath
    #                 try:
    #                     self.config.to_json_file(self.filepath)
    #                     self.notify(f"Project saved to {self.filepath}", severity="success")
    #                     # Update header title in editor
    #                     self.query_one(".editor-title", Static).update(PathLib(self.filepath).name)
    #                     # Update app header
    #                     self._update_header_footer()
    #                 except Exception as e:
    #                     self.notify(f"Error saving project: {e}", severity="error")
            
    #         self.app.push_screen(SaveDialog(self.config), on_dismiss)
    #     else:
    #         try:
    #             self.config.to_json_file(self.filepath)
    #             self.app.current_project = self.config
    #             self.notify(f"Project saved to {self.filepath}", severity="success")
    #         except Exception as e:
    #             self.notify(f"Error saving project: {e}", severity="error")
    
    # def _update_header_footer(self) -> None:
    #     """Update the app's header and footer."""
    #     # Update header title and subtitle
    #     # self.app.update_header("PyCellSegmentation TUI", 
    #     #                       f"Editing: {'New Project' if self.filepath is None else PathLib(self.filepath).name}")
    #     # Footer is updated by app.show_project_editor()
