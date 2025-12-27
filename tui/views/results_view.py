"""Results Explorer view for browsing segmentation results."""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static, Tree, Collapsible
from textual.widgets.tree import TreeNode
from textual.timer import Timer
from rich.text import Text
from rich.syntax import Syntax
from rich.table import Table

from tui.views.base_view import BaseView
from tui.models import ProjectConfig

# Try to import ImageViewer for high-quality image display
try:
    from textual_imageview.viewer import ImageViewer
    HAS_IMAGE_VIEWER = True
except ImportError:
    HAS_IMAGE_VIEWER = False


class ResultsSidebar(Widget):
    """Animated sidebar for the results tree."""
    
    DEFAULT_CSS = """
    ResultsSidebar {
        width: 40;
        layer: sidebar;
        dock: left;
        offset-x: -100%;
        background: $boost;
        border-right: solid $primary;
        transition: offset 200ms;
        
        &.-visible {
            offset-x: 0;
        }
        
        .sidebar-title {
            text-align: center;
            color: $text-muted;
            text-style: bold;
            padding: 1;
            background: $panel;
            border-bottom: solid $primary;
        }
        
        #results-tree {
            height: 1fr;
            padding: 1;
            scrollbar-gutter: stable;
        }
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("Result Tree [t]", classes="sidebar-title")
        yield Tree("Results", id="results-tree")

# Import file paths from src (lightweight import)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.file_paths import RESULTS_DIR_BASE


def format_scale_factor_for_path(scale_factor: float | None) -> str:
    """Formats a scale factor for use in paths/IDs (e.g., _scaled_0_5). Returns empty string if scale is 1.0 or None."""
    if scale_factor is not None and scale_factor != 1.0:
        return f"_scaled_{str(scale_factor).replace('.', '_')}"
    return ""


class ResultsView(BaseView):
    """RESULTS EXPLORER view for browsing segmentation results."""
    
    DEFAULT_CSS = """
    ResultsView {
        layers: sidebar;
    }
    """
    
    BINDINGS = [
        Binding("t", "toggle_tree", "Toggle Tree", tooltip="Show/hide the results tree"),
        Binding("m", "toggle_metadata", "Toggle Metadata", tooltip="Show/hide metadata panel"),
    ]
    
    show_sidebar = reactive(True)
    _loading_spinner_timer: Timer | None = None
    _spinner_frame = 0
    _loading_message = "Loading..."
    
    def compose(self) -> ComposeResult:
        # Animated sidebar with tree
        yield ResultsSidebar()
        
        # Main content area
        with Vertical(id="results-detail-panel"):
            # Collapsible metadata section
            with Collapsible(title="Metadata [m]", id="metadata-collapsible", collapsed=True):
                yield Static("Select an item to view details", id="metadata-content")
            
            # Preview container - can hold either Static or ImageViewer
            yield Vertical(id="results-preview-container")
    
    def on_mount(self) -> None:
        """Called when the view is mounted."""
        self.rebuild_tree()
        # Show sidebar by default
        self.query_one(ResultsSidebar).set_class(self.show_sidebar, "-visible")
    
    def action_toggle_tree(self) -> None:
        """Toggle the tree sidebar visibility."""
        self.show_sidebar = not self.show_sidebar
    
    def watch_show_sidebar(self, show_sidebar: bool) -> None:
        """Update sidebar visibility when reactive changes."""
        self.query_one(ResultsSidebar).set_class(show_sidebar, "-visible")
    
    def action_toggle_metadata(self) -> None:
        """Toggle the metadata panel visibility."""
        collapsible = self.query_one("#metadata-collapsible", Collapsible)
        collapsible.collapsed = not collapsible.collapsed
    
    def rebuild_tree(self) -> None:
        """Rebuild the results tree based on the current config."""
        tree = self.query_one("#results-tree", Tree)
        tree.clear()
        tree.root.expand()
        
        if not self.config:
            tree.root.add_leaf("No config loaded")
            return
        
        # Iterate through active image configurations
        for img_config in self.config.image_configurations:
            if not img_config.is_active:
                continue
            
            image_id = img_config.image_id
            image_node = tree.root.add(f"[bold]{image_id}[/bold]", expand=True)
            image_node.data = {"type": "image", "image_id": image_id}
            
            # Check tiling and scaling settings
            seg_opts = img_config.segmentation_options
            apply_tiling = False
            scale_factor = 1.0
            
            if seg_opts:
                if seg_opts.tiling_parameters and seg_opts.tiling_parameters.apply_tiling:
                    apply_tiling = True
                if seg_opts.rescaling_config and seg_opts.rescaling_config.scale_factor:
                    scale_factor = seg_opts.rescaling_config.scale_factor
            
            # Iterate through active parameter configurations
            for param_config in self.config.cellpose_parameter_configurations:
                if not param_config.is_active:
                    continue
                
                param_set_id = param_config.param_set_id
                param_node = image_node.add(f"{param_set_id}", expand=True)
                param_node.data = {
                    "type": "param_set",
                    "image_id": image_id,
                    "param_set_id": param_set_id
                }
                
                # Construct expected result folder name
                base_id = f"{image_id}_{param_set_id}"
                scale_suffix = format_scale_factor_for_path(scale_factor) if scale_factor != 1.0 else ""
                
                if apply_tiling:
                    # Tiled output - look for folders matching pattern
                    self._add_tiled_results(param_node, base_id, image_id, param_set_id)
                else:
                    # Single output folder
                    folder_name = f"{base_id}{scale_suffix}"
                    self._add_single_results(param_node, folder_name, image_id, param_set_id)
    
    def _add_tiled_results(self, parent_node: TreeNode, base_id: str, image_id: str, param_set_id: str) -> None:
        """Add a summary node for tiled results."""
        if not os.path.exists(RESULTS_DIR_BASE):
            parent_node.add_leaf("[dim]No results directory[/dim]")
            return
        
        # Find all folders starting with base_id_
        matching_folders = []
        for folder in os.listdir(RESULTS_DIR_BASE):
            folder_path = os.path.join(RESULTS_DIR_BASE, folder)
            if os.path.isdir(folder_path) and folder.startswith(f"{base_id}_"):
                matching_folders.append(folder_path)
        
        if not matching_folders:
            parent_node.add_leaf("[dim]No tiled results found[/dim]")
            return
        
        # Count tiles and gather stats
        tile_count = len(matching_folders)
        total_size = 0
        tile_stats = []
        
        for folder_path in sorted(matching_folders):
            folder_name = os.path.basename(folder_path)
            folder_size = 0
            mask_count = 0
            
            for f in os.listdir(folder_path):
                file_path = os.path.join(folder_path, f)
                if os.path.isfile(file_path):
                    folder_size += os.path.getsize(file_path)
                    if f.endswith("_mask.tif"):
                        mask_count += 1
            
            total_size += folder_size
            tile_stats.append({
                "folder": folder_name,
                "path": folder_path,
                "size": folder_size,
                "masks": mask_count
            })
        
        # Add a single summary node
        size_mb = total_size / (1024 * 1024)
        label = f"[cyan]Tiled Output[/cyan] ({tile_count} tiles, {size_mb:.1f} MB)"
        tile_node = parent_node.add_leaf(label)
        tile_node.data = {
            "type": "tiled_output",
            "image_id": image_id,
            "param_set_id": param_set_id,
            "tile_count": tile_count,
            "total_size": total_size,
            "tile_stats": tile_stats
        }
    
    def _add_single_results(self, parent_node: TreeNode, folder_name: str, image_id: str, param_set_id: str) -> None:
        """Add result files from a single (non-tiled) output folder."""
        folder_path = os.path.join(RESULTS_DIR_BASE, folder_name)
        
        if not os.path.exists(folder_path):
            parent_node.add_leaf("[dim]No results found[/dim]")
            return
        
        # List files in the folder
        files = []
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                files.append((f, file_path))
        
        if not files:
            parent_node.add_leaf("[dim]Empty results folder[/dim]")
            return
        
        # Add each file as a leaf node
        for filename, file_path in sorted(files):
            # Determine file type icon
            if filename.endswith("_mask.tif"):
                icon = "[green]mask.tiff[/green]"
                label = f"{icon} ({self._format_size(os.path.getsize(file_path))})"
            elif filename.endswith(".json"):
                icon = "[yellow]summary.json[/yellow]"
                label = f"{icon}"
            elif filename.endswith(".csv"):
                icon = "[blue]stats.csv[/blue]"
                label = f"{icon}"
            else:
                label = filename
            
            file_node = parent_node.add_leaf(label)
            file_node.data = {
                "type": "file",
                "path": file_path,
                "filename": filename,
                "image_id": image_id,
                "param_set_id": param_set_id
            }
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable form."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        node = event.node
        data = node.data
        
        if not data:
            self._clear_preview()
            return
        
        node_type = data.get("type")
        
        if node_type == "image":
            self._show_image_summary(data)
        elif node_type == "param_set":
            self._show_param_summary(data)
        elif node_type == "tiled_output":
            self._show_tiled_summary(data)
        elif node_type == "file":
            # Use call_later to ensure async methods can be called
            self.call_later(self._show_file_preview, data)
        else:
            self._clear_preview()
    
    def _clear_preview(self) -> None:
        """Clear the preview pane."""
        metadata = self.query_one("#metadata-content", Static)
        metadata.update("Select an item to view details")
        self._show_text_preview("")
    
    def _clear_preview_container(self) -> None:
        """Safely clear the preview container."""
        # Stop spinner timer if running
        if self._loading_spinner_timer:
            self._loading_spinner_timer.stop()
            self._loading_spinner_timer = None
        
        try:
            container = self.query_one("#results-preview-container", Vertical)
            # Try to remove ImageViewer widgets first if they exist
            if HAS_IMAGE_VIEWER:
                try:
                    existing_viewers = container.query(ImageViewer)
                    for viewer in existing_viewers:
                        try:
                            viewer.remove()
                        except Exception:
                            pass
                except Exception:
                    pass
            
            # Remove all children
            try:
                container.remove_children()
            except Exception:
                # Fallback: try removing children one by one
                try:
                    for child in list(container.children):
                        child.remove()
                except Exception:
                    pass
        except Exception:
            # If we can't find the container, that's okay - it might not exist yet
            pass
    
    def _show_text_preview(self, content) -> None:
        """Show text/renderable content in the preview container."""
        self._clear_preview_container()
        
        try:
            container = self.query_one("#results-preview-container", Vertical)
            # Create a scrollable container with Static for text content
            scroll = ScrollableContainer(id="results-preview")
            static = Static(content, id="preview-content")
            container.mount(scroll)
            scroll.mount(static)
        except Exception as e:
            # If mounting fails, at least try to show error
            try:
                container = self.query_one("#results-preview-container", Vertical)
                error_static = Static(f"[red]Error displaying preview: {e}[/red]", id="preview-error")
                container.mount(error_static)
            except Exception:
                pass
    
    def _show_image_preview(self, image_path: str) -> None:
        """Show an image using ImageViewer in the preview container."""
        self._clear_preview_container()
        
        if HAS_IMAGE_VIEWER:
            try:
                from PIL import Image
                # Load and display using ImageViewer
                image = Image.open(image_path)
                container = self.query_one("#results-preview-container", Vertical)
                viewer = ImageViewer(image, id="image-viewer")
                container.mount(viewer)
                return
            except Exception as e:
                # Fall back to text preview on error
                self._show_text_preview(f"[red]Error loading image: {e}[/red]")
                return
        
        # Fallback if ImageViewer not available
        self._show_text_preview("[dim]ImageViewer not available. Install textual-imageview for image preview.[/dim]")
    
    def _show_image_summary(self, data: dict) -> None:
        """Show summary for an image node."""
        image_id = data.get("image_id", "Unknown")
        metadata = self.query_one("#metadata-content", Static)
        
        metadata.update(f"[bold]Image:[/bold] {image_id}")
        self._show_text_preview("Select a parameter set or result file for details.")
    
    def _show_param_summary(self, data: dict) -> None:
        """Show summary for a parameter set node."""
        image_id = data.get("image_id", "Unknown")
        param_set_id = data.get("param_set_id", "Unknown")
        
        metadata = self.query_one("#metadata-content", Static)
        
        metadata.update(f"[bold]Image:[/bold] {image_id}\n[bold]Params:[/bold] {param_set_id}")
        self._show_text_preview("Select a result file for details.")
    
    def _show_tiled_summary(self, data: dict) -> None:
        """Show summary for tiled output."""
        metadata = self.query_one("#metadata-content", Static)
        
        tile_count = data.get("tile_count", 0)
        total_size = data.get("total_size", 0)
        tile_stats = data.get("tile_stats", [])
        
        # Metadata
        size_mb = total_size / (1024 * 1024)
        meta_text = (
            f"[bold]Total Tiles:[/bold] {tile_count}\n"
            f"[bold]Total Size:[/bold] {size_mb:.2f} MB\n"
            f"[bold]Image:[/bold] {data.get('image_id', 'N/A')}\n"
            f"[bold]Params:[/bold] {data.get('param_set_id', 'N/A')}"
        )
        metadata.update(meta_text)
        
        # Preview - show a table of tile stats
        table = Table(title="Tile Summary", show_header=True, header_style="bold cyan")
        table.add_column("Tile", style="dim")
        table.add_column("Masks", justify="right")
        table.add_column("Size", justify="right")
        
        # Limit to first 20 tiles to avoid overwhelming the UI
        for i, stat in enumerate(tile_stats[:20]):
            folder = stat.get("folder", "")
            # Extract tile identifier from folder name
            tile_id = folder.split("_")[-1] if "_" in folder else folder
            masks = stat.get("masks", 0)
            size = self._format_size(stat.get("size", 0))
            table.add_row(tile_id, str(masks), size)
        
        if len(tile_stats) > 20:
            table.add_row(f"... and {len(tile_stats) - 20} more", "", "")
        
        self._show_text_preview(table)
    
    def _show_file_preview(self, data: dict) -> None:
        """Show preview for a single file."""
        file_path = data.get("path", "")
        filename = data.get("filename", "")
        
        metadata = self.query_one("#metadata-content", Static)
        
        if not os.path.exists(file_path):
            metadata.update(f"[bold]File:[/bold] {filename}\n[red]File not found[/red]")
            self._show_text_preview("")
            return
        
        # Get file stats
        stat = os.stat(file_path)
        size = self._format_size(stat.st_size)
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        meta_text = (
            f"[bold]File:[/bold] {filename}\n"
            f"[bold]Size:[/bold] {size}\n"
            f"[bold]Modified:[/bold] {modified}\n"
            f"[bold]Path:[/bold] {file_path}"
        )
        metadata.update(meta_text)
        
        # Preview content based on file type
        if filename.endswith(".json"):
            self._preview_json(file_path)
        elif filename.endswith(".csv"):
            self._preview_csv(file_path)
        elif filename.endswith((".tif", ".tiff")):
            self._preview_tiff(file_path)
        else:
            self._show_text_preview("[dim]Preview not available for this file type[/dim]")
    
    def _preview_json(self, file_path: str) -> None:
        """Preview JSON file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(4096)  # Limit to first 4KB
            
            # Truncate if too long
            if len(content) >= 4096:
                content = content[:4000] + "\n... (truncated)"
            
            syntax = Syntax(content, "json", theme="monokai", line_numbers=True)
            self._show_text_preview(syntax)
        except Exception as e:
            self._show_text_preview(f"[red]Error reading file: {e}[/red]")
    
    def _preview_csv(self, file_path: str) -> None:
        """Preview CSV file content."""
        try:
            lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 20:  # Limit to first 20 lines
                        lines.append("... (truncated)")
                        break
                    lines.append(line.rstrip())
            
            content = "\n".join(lines)
            syntax = Syntax(content, "csv", theme="monokai", line_numbers=True)
            self._show_text_preview(syntax)
        except Exception as e:
            self._show_text_preview(f"[red]Error reading file: {e}[/red]")
    
    def _preview_tiff(self, file_path: str) -> None:
        """Preview TIFF file using ImageViewer for masks."""
        # For mask files, use ImageViewer with colorized mask
        if "_mask" in file_path:
            # Load asynchronously to avoid blocking UI
            # Use call_later to schedule the async function properly
            self.app.call_later(self._start_mask_loading, file_path)
            return
    
    def _start_mask_loading(self, file_path: str) -> None:
        """Start the async mask loading task."""
        import asyncio
        try:
            # Get the running event loop (Textual apps run in an async context)
            loop = asyncio.get_running_loop()
            loop.create_task(self._preview_mask_image_async(file_path))
        except RuntimeError:
            # Fallback: try to get event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._preview_mask_image_async(file_path))
                else:
                    loop.run_until_complete(self._preview_mask_image_async(file_path))
            except Exception as e:
                # Last resort: show error
                self._show_text_preview(f"[red]Error starting mask load: {e}[/red]")
        
        # For other TIFF files, show metadata
        try:
            import tifffile
            
            with tifffile.TiffFile(file_path) as tif:
                page = tif.pages[0]
                shape = page.shape
                dtype = page.dtype
                
                info_text = (
                    f"[bold]Image Type:[/bold] TIFF\n"
                    f"[bold]Dimensions:[/bold] {shape}\n"
                    f"[bold]Data Type:[/bold] {dtype}\n"
                    f"[bold]Pages:[/bold] {len(tif.pages)}"
                )
                self._show_text_preview(info_text)
                
        except ImportError as e:
            self._show_text_preview(f"[dim]Missing dependency: {e}[/dim]")
        except Exception as e:
            self._show_text_preview(f"[red]Error reading TIFF: {e}[/red]")
    
    def _show_loading_indicator(self, message: str = "Loading...") -> None:
        """Show a loading indicator with animated spinner in the preview container."""
        self._clear_preview_container()
        try:
            container = self.query_one("#results-preview-container", Vertical)
            # Store message for spinner updates
            self._loading_message = message
            loading = Static(f"[dim]{message} ⠋[/dim]", id="loading-indicator")
            loading.styles.align = ("center", "middle")
            container.mount(loading)
            
            # Start spinner animation
            self._spinner_frame = 0
            if self._loading_spinner_timer:
                self._loading_spinner_timer.stop()
            self._loading_spinner_timer = self.set_interval(0.1, self._update_spinner)
        except Exception:
            pass
    
    def _update_spinner(self) -> None:
        """Update the spinner animation."""
        try:
            loading = self.query_one("#loading-indicator", Static)
            # Braille spinner frames: ⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏
            spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            frame = spinner_frames[self._spinner_frame % len(spinner_frames)]
            loading.update(f"[dim]{self._loading_message} {frame}[/dim]")
            self._spinner_frame += 1
        except Exception:
            # Stop timer if widget is gone
            if self._loading_spinner_timer:
                self._loading_spinner_timer.stop()
                self._loading_spinner_timer = None
    
    async def _preview_mask_image_async(self, file_path: str) -> None:
        """Preview mask TIFF using ImageViewer asynchronously."""
        # Show loading indicator immediately
        self._show_loading_indicator("Loading mask...")
        
        # Process in background to avoid blocking UI
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def process_mask():
            """Process mask in background thread."""
            import tifffile
            import numpy as np
            from PIL import Image
            
            # Load the mask
            with tifffile.TiffFile(file_path) as tif:
                data = tif.pages[0].asarray()
            
            # For faster loading, skip colorization for very large masks
            # Just convert to grayscale RGB for display
            h, w = data.shape[:2]
            total_pixels = h * w
            
            # If mask is very large (> 10M pixels), use simpler rendering
            if total_pixels > 10_000_000:
                # Simple grayscale conversion for speed
                # Normalize to 0-255
                normalized = ((data.astype(np.float32) / data.max()) * 255).astype(np.uint8)
                rgb_array = np.stack([normalized, normalized, normalized], axis=-1)
            else:
                # Colorize the mask for smaller images
                rgb_array = self._colorize_mask(data)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_array, mode='RGB')
            
            return pil_image, data
        
        try:
            # Run processing in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                pil_image, data = await loop.run_in_executor(executor, process_mask)
            
            # Update UI on the main thread - use call_later to ensure it runs on UI thread
            # This ensures the spinner stays visible until the image is actually displayed
            self.call_later(self._display_mask_image, pil_image, data, file_path)
            
        except Exception as e:
            # Stop spinner on error
            if self._loading_spinner_timer:
                self._loading_spinner_timer.stop()
                self._loading_spinner_timer = None
            
            # Show error or fallback
            error_msg = f"[red]Error loading mask: {e}[/red]"
            try:
                self._show_text_preview(error_msg)
            except Exception:
                # If showing error fails, try fallback
                try:
                    # Try to get data for fallback if we have it
                    if 'data' in locals():
                        self._preview_tiff_fallback(data)
                    else:
                        self._show_text_preview("[red]Failed to load mask[/red]")
                except Exception:
                    pass
    
    def _display_mask_image(self, pil_image, data, file_path: str) -> None:
        """Display the processed mask image in the UI."""
        # Stop spinner before displaying
        if self._loading_spinner_timer:
            self._loading_spinner_timer.stop()
            self._loading_spinner_timer = None
        
        # Use ImageViewer if available
        if HAS_IMAGE_VIEWER:
            try:
                # Clear container but don't use _clear_preview_container as it stops spinner
                container = self.query_one("#results-preview-container", Vertical)
                
                # Remove existing children (including loading indicator)
                try:
                    # Try to remove ImageViewer widgets first if they exist
                    if HAS_IMAGE_VIEWER:
                        try:
                            existing_viewers = container.query(ImageViewer)
                            for viewer in existing_viewers:
                                try:
                                    viewer.remove()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    
                    # Remove all children
                    try:
                        container.remove_children()
                    except Exception:
                        # Fallback: try removing children one by one
                        try:
                            for child in list(container.children):
                                child.remove()
                        except Exception:
                            pass
                except Exception:
                    pass
                
                # Create and mount new viewer
                viewer = ImageViewer(pil_image, id="mask-viewer")
                container.mount(viewer)
                
                # Image is now displayed
            except Exception as e:
                # Fall back to text preview on error
                error_msg = f"[red]Error displaying mask: {e}[/red]"
                try:
                    self._show_text_preview(error_msg)
                except Exception:
                    pass
                try:
                    self._preview_tiff_fallback(data)
                except Exception:
                    pass
        else:
            # Fallback to text-based rendering
            try:
                self._preview_tiff_fallback(data)
            except Exception:
                pass
    
    def _preview_tiff_fallback(self, mask_data) -> None:
        """Fallback preview using Unicode block rendering."""
        import numpy as np
        
        unique_vals = np.unique(mask_data)
        cell_count = len(unique_vals) - 1 if 0 in unique_vals else len(unique_vals)
        
        info_parts = [
            f"[bold]Cell Count:[/bold] {cell_count}",
            "",
            "[bold]Visual Preview:[/bold] (Install textual-imageview for interactive view)",
            "",
        ]
        
        visual = self._render_mask_fallback(mask_data, max_width=80, max_height=30)
        info_parts.append(visual)
        
        self._show_text_preview("\n".join(info_parts))
    
    def _colorize_mask(self, mask_data):
        """Convert mask labels to RGB image with distinct colors."""
        import numpy as np
        
        h, w = mask_data.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background is dark
        rgb[:, :] = [20, 20, 20]
        
        unique_ids = np.unique(mask_data)
        unique_ids = unique_ids[unique_ids > 0]
        
        # For performance: limit colorization to first 1000 cells
        # This prevents slowdown on masks with many cells
        if len(unique_ids) > 1000:
            unique_ids = unique_ids[:1000]
        
        # Generate colors using golden ratio for good distribution
        colors = self._generate_colors(len(unique_ids))
        
        # Use vectorized operations for better performance
        for i, cell_id in enumerate(unique_ids):
            mask = mask_data == cell_id
            rgb[mask] = colors[i % len(colors)]
        
        return rgb
    
    def _render_mask_fallback(self, mask_data, max_width: int = 80, max_height: int = 40) -> str:
        """Fallback: Render mask as colorized Unicode blocks."""
        import numpy as np
        
        h, w = mask_data.shape[:2]
        
        # Calculate scale factor
        scale_h = max(1, h // (max_height * 2))
        scale_w = max(1, w // max_width)
        scale = max(scale_h, scale_w)
        
        downsampled = mask_data[::scale, ::scale]
        dh, dw = downsampled.shape[:2]
        
        if dh > max_height * 2:
            downsampled = downsampled[:max_height * 2, :]
            dh = max_height * 2
        if dw > max_width:
            downsampled = downsampled[:, :max_width]
            dw = max_width
        
        unique_ids = np.unique(downsampled)
        unique_ids = unique_ids[unique_ids > 0]
        
        colors = self._generate_colors(len(unique_ids) + 1)
        id_to_color = {0: (30, 30, 30)}
        for i, cell_id in enumerate(unique_ids):
            id_to_color[cell_id] = colors[i % len(colors)]
        
        lines = []
        for row in range(0, dh - 1, 2):
            line_chars = []
            for col in range(dw):
                top_id = downsampled[row, col]
                bot_id = downsampled[row + 1, col] if row + 1 < dh else 0
                
                tr, tg, tb = id_to_color.get(top_id, (100, 100, 100))
                br, bg, bb = id_to_color.get(bot_id, (100, 100, 100))
                
                char = f"[rgb({tr},{tg},{tb}) on rgb({br},{bg},{bb})]▀[/]"
                line_chars.append(char)
            
            lines.append("".join(line_chars))
        
        if dh % 2 == 1:
            line_chars = []
            for col in range(dw):
                cell_id = downsampled[dh - 1, col]
                r, g, b = id_to_color.get(cell_id, (100, 100, 100))
                char = f"[rgb({r},{g},{b}) on rgb(30,30,30)]▀[/]"
                line_chars.append(char)
            lines.append("".join(line_chars))
        
        return "\n".join(lines)
    
    def _generate_colors(self, n: int) -> list:
        """Generate n distinct colors for cell visualization."""
        import colorsys
        
        base_colors = [
            (255, 99, 71),    # Tomato
            (50, 205, 50),    # Lime green
            (30, 144, 255),   # Dodger blue
            (255, 215, 0),    # Gold
            (238, 130, 238),  # Violet
            (0, 206, 209),    # Dark turquoise
            (255, 140, 0),    # Dark orange
            (147, 112, 219),  # Medium purple
            (60, 179, 113),   # Medium sea green
            (255, 105, 180),  # Hot pink
            (100, 149, 237),  # Cornflower blue
            (240, 128, 128),  # Light coral
            (144, 238, 144),  # Light green
            (173, 216, 230),  # Light blue
            (255, 182, 193),  # Light pink
            (255, 255, 102),  # Light yellow
        ]
        
        if n <= len(base_colors):
            return base_colors[:n]
        
        colors = list(base_colors)
        for i in range(len(base_colors), n):
            hue = (i * 0.618033988749895) % 1.0
            sat = 0.7 + (i % 3) * 0.1
            val = 0.8 + (i % 2) * 0.1
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        
        return colors
