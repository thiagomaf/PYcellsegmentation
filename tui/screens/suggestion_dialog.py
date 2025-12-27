"""Suggestion dialog modal screen for parameter suggestions."""
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Button, Static, DataTable

from typing import List, Dict, Optional


class SuggestionDialog(ModalScreen[Optional[List[Dict[str, float]]]]):
    """Modal screen for displaying and selecting parameter suggestions.
    
    Returns:
        List of selected parameter dictionaries (for batch config generation),
        or None if cancelled
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "toggle_selection", "Toggle Selection"),
        ("space", "toggle_selection", "Toggle Selection"),
    ]
    
    def __init__(self, suggestions: List[Dict[str, float]], rationales: Optional[List[str]] = None):
        """Initialize the suggestion dialog.
        
        Args:
            suggestions: List of parameter dictionaries
            rationales: Optional list of rationale strings for each suggestion
        """
        super().__init__()
        self.suggestions = suggestions
        self.rationales = rationales or [""] * len(suggestions)
        self.selected_indices = set()  # Multi-select support
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the suggestion dialog."""
        with Container(classes="suggestion-dialog-container"):
            yield Static("Suggested Parameter Sets (Select multiple for batch config)", classes="suggestion-dialog-header")
            
            with ScrollableContainer(classes="suggestion-dialog-list"):
                yield DataTable(id="suggestion-table")
            
            with Horizontal(classes="suggestion-dialog-footer"):
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Create Config (Selected)", id="create", variant="primary")
                yield Static("", id="selection-count")
    
    def on_mount(self) -> None:
        """Initialize the table with suggestions."""
        table = self.query_one("#suggestion-table", DataTable)
        table.add_columns("✓", "Diameter", "Min Size", "Flow", "CellProb", "Rationale")
        table.cursor_type = "row"  # Enable row selection
        
        for i, (params, rationale) in enumerate(zip(self.suggestions, self.rationales)):
            table.add_row(
                "○",
                f"{params.get('diameter', 0):.1f}",
                f"{params.get('min_size', 0)}",
                f"{params.get('flow_threshold', 0):.2f}",
                f"{params.get('cellprob_threshold', 0):.2f}",
                rationale[:60] + "..." if len(rationale) > 60 else rationale
            )
        
        # Update selection count
        self._update_selection_count()
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection (toggle multi-select)."""
        row_idx = event.cursor_row
        
        # Toggle selection
        if row_idx in self.selected_indices:
            self.selected_indices.remove(row_idx)
            self.query_one("#suggestion-table", DataTable).update_cell_at((row_idx, 0), "○")
        else:
            self.selected_indices.add(row_idx)
            self.query_one("#suggestion-table", DataTable).update_cell_at((row_idx, 0), "✓")
        
        self._update_selection_count()
    
    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Handle cell selection - also triggers row selection toggle."""
        row_idx = event.coordinate.row
        
        # Toggle selection
        if row_idx in self.selected_indices:
            self.selected_indices.remove(row_idx)
            self.query_one("#suggestion-table", DataTable).update_cell_at((row_idx, 0), "○")
        else:
            self.selected_indices.add(row_idx)
            self.query_one("#suggestion-table", DataTable).update_cell_at((row_idx, 0), "✓")
        
        self._update_selection_count()
    
    def _update_selection_count(self) -> None:
        """Update the selection count display."""
        count_widget = self.query_one("#selection-count", Static)
        count = len(self.selected_indices)
        if count == 0:
            count_widget.update("No selection")
        elif count == 1:
            count_widget.update("1 selected")
        else:
            count_widget.update(f"{count} selected")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "create":
            if self.selected_indices:
                # Return all selected suggestions
                selected = [self.suggestions[i] for i in sorted(self.selected_indices)]
                self.dismiss(selected)
            else:
                # If no selection, use first suggestion
                if self.suggestions:
                    self.dismiss([self.suggestions[0]])
                else:
                    self.dismiss(None)
    
    def action_cancel(self) -> None:
        """Cancel action."""
        self.dismiss(None)
    
    def action_toggle_selection(self) -> None:
        """Toggle selection of current row."""
        table = self.query_one("#suggestion-table", DataTable)
        row_idx = table.cursor_row
        
        if row_idx is not None and 0 <= row_idx < len(self.suggestions):
            # Toggle selection
            if row_idx in self.selected_indices:
                self.selected_indices.remove(row_idx)
                table.update_cell_at((row_idx, 0), "○")
            else:
                self.selected_indices.add(row_idx)
                table.update_cell_at((row_idx, 0), "✓")
            
            self._update_selection_count()


