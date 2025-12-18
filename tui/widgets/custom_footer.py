"""Custom Footer widget that can filter bindings dynamically."""
from typing import Optional
from textual.binding import Binding
from textual.widgets import Footer
from textual.strip import Strip
from rich.segment import Segment
from rich.style import Style


class CustomFooter(Footer):
    """A Footer widget that can display filtered bindings.
    
    This widget extends Textual's Footer to allow filtering which bindings
    are displayed based on a provided list of keys and descriptions.
    """
    
    def __init__(
        self,
        visible_bindings: Optional[list[tuple[str, str]]] = None,
        **kwargs
    ) -> None:
        """Initialize the custom footer.
        
        Args:
            visible_bindings: Optional list of (key, description) tuples to show.
                If None, shows all bindings from app/screen.
            **kwargs: Additional arguments passed to Footer.
        """
        super().__init__(**kwargs)
        self._visible_bindings: Optional[list[tuple[str, str]]] = visible_bindings
    
    def set_bindings(self, bindings: Optional[list[tuple[str, str]]]) -> None:
        """Set which bindings should be visible.
        
        Args:
            bindings: List of (key, description) tuples to show, or None to show all bindings.
        """
        self._visible_bindings = bindings
        self.refresh()
    
    def _get_filtered_bindings(self) -> list[Binding]:
        """Get bindings filtered by visible_bindings.
        
        Returns:
            List of Binding objects to display.
        """
        # Get bindings from app/screen directly
        all_bindings = []
        screen = self.screen
        app = self.app
        
        # Try to get bindings from screen first, then app
        source_bindings = None
        if screen and hasattr(screen, 'BINDINGS'):
            source_bindings = screen.BINDINGS
        elif app and hasattr(app, 'BINDINGS'):
            source_bindings = app.BINDINGS
        
        if source_bindings:
            for binding in source_bindings:
                if isinstance(binding, Binding):
                    all_bindings.append(binding)
                elif isinstance(binding, tuple) and len(binding) >= 3:
                    # Convert tuple to Binding: (key, action, description)
                    all_bindings.append(Binding(binding[0], binding[1], binding[2], show=True))
        
        # If we don't have a custom filter, return all bindings
        if self._visible_bindings is None:
            return [b for b in all_bindings if b.show]
        
        # Create a mapping of keys to our custom descriptions
        visible_keys = {key for key, _ in self._visible_bindings}
        custom_descriptions = {key: desc for key, desc in self._visible_bindings}
        
        # Filter and create new Binding objects with custom descriptions
        filtered = []
        for binding in all_bindings:
            if binding.key in visible_keys and binding.show:
                # Use custom description if provided, otherwise use original
                description = custom_descriptions.get(binding.key, binding.description)
                filtered.append(Binding(binding.key, binding.action, description, show=True))
        
        return filtered
    
    def render_line(self, y: int) -> Strip:
        """Override render_line to display filtered bindings.
        
        Args:
            y: Line number (should be 0 for footer).
            
        Returns:
            Strip object with the footer content.
        """
        try:
            # Get filtered bindings
            bindings = self._get_filtered_bindings()
            
            if not bindings:
                # Return empty strip if no bindings
                return Strip([Segment(" " * self.size.width)])
            
            # Get styles
            key_style = self.get_component_rich_style("footer--key")
            desc_style = self.get_component_rich_style("footer--description")
            bar_style = self.get_component_rich_style("footer--bar")
            
            # Format bindings as segments
            segments = []
            for binding in bindings:
                # Use make_key_display if available, otherwise format manually
                if hasattr(self, 'make_key_display'):
                    key_display = self.make_key_display(binding.key)
                else:
                    key_display = self._format_key(binding.key)
                
                segments.append(Segment(f" {key_display} ", key_style))
                segments.append(Segment(f" {binding.description} ", desc_style))
                segments.append(Segment("  ", bar_style))
            
            # Create a strip with the segments
            # Pad with bar style to full width
            strip = Strip(segments)
            
            # Calculate remaining width and pad
            width = self.size.width
            current_width = strip.cell_length
            if current_width < width:
                segments.append(Segment(" " * (width - current_width), bar_style))
                strip = Strip(segments)
            
            return strip.crop(0, width)
            
        except Exception:
            # Fallback in case of error to prevent crash
            return Strip([Segment("Error rendering footer")])
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        super().on_mount()
        self.refresh()
    
    def _format_key(self, key: str) -> str:
        """Format a key for display in the footer.
        
        Args:
            key: The key binding (e.g., "ctrl+s", "escape", "q")
            
        Returns:
            Formatted key string for display.
        """
        # Handle modifier keys
        parts = key.split("+")
        if len(parts) > 1:
            # Has modifiers like "ctrl+s"
            modifiers = parts[:-1]
            main_key = parts[-1]
            formatted_modifiers = [mod.capitalize() for mod in modifiers]
            return f"{'+'.join(formatted_modifiers)}+{main_key.upper()}"
        else:
            # Simple key
            if len(key) == 1:
                return key.upper()
            elif key == "escape":
                return "Esc"
            elif key.startswith("f") and key[1:].isdigit():
                return key.upper()  # Function keys like F12
            else:
                return key.capitalize()