"""Simple inspector screen as fallback when textual run --dev is not used."""
from textual.screen import ModalScreen
from textual.widgets import Static, Tree, Input, Button
from textual.containers import Container, Vertical, Horizontal
from textual import events


class SimpleInspector(ModalScreen):
    """A simple inspector that shows widget tree and properties."""
    
    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]
    
    def __init__(self):
        """Initialize the inspector."""
        super().__init__()
        self._target_screen = None
    
    def compose(self):
        """Compose the inspector UI."""
        with Container(classes="inspector-container"):
            with Container(classes="inspector-header"):
                yield Static("Widget Inspector (Simple Mode)", classes="inspector-title")
                yield Static("Press Esc to close | / to search", classes="inspector-hint")
            
            with Horizontal(classes="inspector-content"):
                with Vertical(classes="tree-container"):
                    yield Static("Widget Tree (Current Screen):", classes="section-title")
                    yield Input(placeholder="Search by ID or class (e.g., nav-general, nav-btn)...", id="search-input")
                    yield Tree("Widgets", id="widget-tree")
                
                with Container(classes="details-container"):
                    yield Static("Widget Details:", classes="section-title")
                    yield Static("Select a widget from the tree to see details", id="widget-details")
    
    def on_mount(self):
        """Build the widget tree when mounted."""
        tree = self.query_one("#widget-tree", Tree)
        
        # Get the screen that was active before the inspector was opened
        target_screen = None
        
        # First, try to use the stored target screen (set by app when pushing)
        if hasattr(self, '_target_screen') and self._target_screen is not None:
            target_screen = self._target_screen
        
        # If we still don't have a target, try getting it from the app
        if target_screen is None:
            # Try to access the screen stack - when a modal is pushed,
            # the previous screen should be accessible
            try:
                # Textual stores screens in _screen_stack
                if hasattr(self.app, '_screen_stack'):
                    stack = self.app._screen_stack
                    # Find this inspector in the stack and get the one before it
                    for i, screen in enumerate(stack):
                        if screen is self and i > 0:
                            target_screen = stack[i - 1]
                            break
            except:
                pass
        
        # Last resort: use the app itself (shows all screens)
        if target_screen is None:
            target_screen = self.app
        
        self._build_tree(tree, target_screen, None)
        tree.root.expand()
        
        # Set up search - don't focus immediately to avoid stealing focus
        # search_input = self.query_one("#search-input", Input)
        # search_input.focus()
    
    def _build_tree(self, tree, widget, parent_node):
        """Recursively build the widget tree."""
        widget_id = widget.id or "(no id)"
        widget_type = type(widget).__name__
        classes_str = ""
        try:
            classes = getattr(widget, 'classes', set())
            if classes:
                classes_str = f" .{'.'.join(classes)}"
        except:
            pass
        
        # Create label with type, ID, and classes
        if widget.id:
            label = f"{widget_type}#{widget_id}{classes_str}"
        else:
            label = f"{widget_type}{classes_str}"
        
        if parent_node is None:
            node = tree.root.add(label, data=widget)
        else:
            node = parent_node.add(label, data=widget)
        
        # Add children
        for child in widget.children:
            self._build_tree(tree, child, node)
    
    def on_tree_node_selected(self, event: Tree.NodeSelected):
        """Show details when a widget is selected."""
        try:
            widget = event.node.data
            if widget is None:
                return
            
            details = self._get_widget_details(widget)
            details_widget = self.query_one("#widget-details", Static)
            details_widget.update(details)
        except Exception as e:
            # Show error in details widget instead of crashing
            try:
                details_widget = self.query_one("#widget-details", Static)
                details_widget.update(f"Error getting widget details:\n{type(e).__name__}: {e}")
            except:
                # If we can't even update the widget, silently fail
                pass
    
    def _get_widget_details(self, widget):
        """Get formatted details for a widget."""
        lines = []
        
        # Safely get basic properties
        try:
            lines.append(f"Type: {type(widget).__name__}")
        except Exception as e:
            lines.append(f"Type: (error: {e})")
        
        try:
            widget_id = getattr(widget, 'id', None) or '(no id)'
            lines.append(f"ID: {widget_id}")
            if widget_id != '(no id)':
                lines.append(f"  CSS Selector: #{widget_id}")
        except Exception as e:
            lines.append(f"ID: (error: {e})")
        
        try:
            classes = getattr(widget, 'classes', set())
            if classes:
                classes_str = ', '.join(classes)
                lines.append(f"Classes: {classes_str}")
                for cls in classes:
                    lines.append(f"  CSS Selector: .{cls}")
            else:
                lines.append("Classes: (none)")
        except Exception as e:
            lines.append(f"Classes: (error: {e})")
        
        try:
            lines.append(f"Size: {getattr(widget, 'size', 'unknown')}")
        except Exception as e:
            lines.append(f"Size: (error: {e})")
        
        try:
            lines.append(f"Region: {getattr(widget, 'region', 'unknown')}")
        except Exception as e:
            lines.append(f"Region: (error: {e})")
        
        try:
            lines.append(f"Visible: {getattr(widget, 'visible', 'unknown')}")
        except Exception as e:
            lines.append(f"Visible: (error: {e})")
        
        try:
            lines.append(f"Disabled: {getattr(widget, 'disabled', 'unknown')}")
        except Exception as e:
            lines.append(f"Disabled: (error: {e})")
        
        lines.append("")
        lines.append("CSS Styles (Computed):")
        
        # Get computed styles with error handling
        try:
            styles = getattr(widget, 'styles', None)
            if styles:
                style_props = [
                    ('background', 'Background'),
                    ('color', 'Color'),
                    ('border', 'Border'),
                    ('border-top', 'Border Top'),
                    ('border-right', 'Border Right'),
                    ('border-bottom', 'Border Bottom'),
                    ('border-left', 'Border Left'),
                    ('padding', 'Padding'),
                    ('padding-top', 'Padding Top'),
                    ('padding-right', 'Padding Right'),
                    ('padding-bottom', 'Padding Bottom'),
                    ('padding-left', 'Padding Left'),
                    ('margin', 'Margin'),
                    ('margin-top', 'Margin Top'),
                    ('margin-right', 'Margin Right'),
                    ('margin-bottom', 'Margin Bottom'),
                    ('margin-left', 'Margin Left'),
                    ('width', 'Width'),
                    ('height', 'Height'),
                    ('text-align', 'Text Align'),
                    ('text-style', 'Text Style'),
                    ('opacity', 'Opacity'),
                    ('display', 'Display'),
                ]
                
                for prop, label in style_props:
                    try:
                        value = getattr(styles, prop, None)
                        if value is not None and str(value) != 'None':
                            lines.append(f"  {label}: {value}")
                    except:
                        pass
            else:
                lines.append("  (No styles object)")
        except Exception as e:
            lines.append(f"  (Error getting styles: {e})")
        
        return "\n".join(lines)
    
    def on_input_changed(self, event: Input.Changed):
        """Handle search input changes."""
        if event.input.id != "search-input":
            return
        
        search_term = event.value.lower().strip()
        tree = self.query_one("#widget-tree", Tree)
        
        if not search_term:
            # Reset - expand root
            tree.root.expand()
            return
        
        # Recursively search through tree nodes
        def search_nodes(node, term, matches):
            """Recursively search tree nodes."""
            widget = node.data if hasattr(node, 'data') else None
            if widget:
                # Check ID
                widget_id = getattr(widget, 'id', None) or ''
                if term in widget_id.lower():
                    matches.append(node)
                
                # Check classes
                try:
                    classes = getattr(widget, 'classes', set())
                    for cls in classes:
                        if term in cls.lower():
                            matches.append(node)
                            break
                except:
                    pass
                
                # Check type
                widget_type = type(widget).__name__
                if term in widget_type.lower():
                    matches.append(node)
            
            # Search children
            if hasattr(node, 'children'):
                for child in node.children:
                    search_nodes(child, term, matches)
        
        matching_nodes = []
        search_nodes(tree.root, search_term, matching_nodes)
        
        # Expand matching nodes and their parents
        for node in matching_nodes:
            # Expand this node and all its parents
            current = node
            while current and hasattr(current, 'expand'):
                current.expand()
                current = getattr(current, 'parent', None)
    
    def action_dismiss(self):
        """Close the inspector."""
        self.dismiss()

