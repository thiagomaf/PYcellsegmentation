"""
Napari Widget for Layer Color Control
======================================

A custom widget that allows users to change mask layer colors interactively.
"""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QColorDialog, QLabel, QCheckBox, QSlider
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
import napari
import numpy as np


class LayerColorControl(QWidget):
    """Widget for controlling label layer colors."""
    
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.current_layer = None
        self.setup_ui()
        
        # Connect to layer selection changes
        self.viewer.layers.events.inserted.connect(self._update_layer_list)
        self.viewer.layers.events.removed.connect(self._update_layer_list)
        
    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QLabel("Layer Color Control")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Layer selection
        layout.addWidget(QLabel("Select Layer:"))
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self._on_layer_selected)
        layout.addWidget(self.layer_combo)
        
        # Color mode toggle
        self.single_color_check = QCheckBox("Single Color Mode")
        self.single_color_check.stateChanged.connect(self._on_color_mode_changed)
        layout.addWidget(self.single_color_check)
        
        # Color picker button
        self.color_button = QPushButton("Choose Color")
        self.color_button.clicked.connect(self._pick_color)
        self.color_button.setEnabled(False)
        layout.addWidget(self.color_button)
        
        # Current color display
        self.color_label = QLabel("Current: Multi-color")
        self.color_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
        layout.addWidget(self.color_label)
        
        # Opacity control
        layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel("70%")
        layout.addWidget(self.opacity_label)
        
        # Shuffle colors button (for multi-color mode)
        self.shuffle_button = QPushButton("Shuffle Colors")
        self.shuffle_button.clicked.connect(self._shuffle_colors)
        layout.addWidget(self.shuffle_button)
        
        layout.addStretch()
        self._update_layer_list()
        
    def _update_layer_list(self):
        """Update the layer dropdown with current labels layers."""
        self.layer_combo.clear()
        from napari.layers import Labels
        labels_layers = [layer for layer in self.viewer.layers if isinstance(layer, Labels)]
        
        for layer in labels_layers:
            self.layer_combo.addItem(layer.name, layer)
            
        if labels_layers:
            self.layer_combo.setCurrentIndex(0)
            self._on_layer_selected(self.layer_combo.currentText())
        else:
            self.current_layer = None
            self.color_button.setEnabled(False)
            
    def _on_layer_selected(self, layer_name):
        """Handle layer selection."""
        from napari.layers import Labels
        labels_layers = [layer for layer in self.viewer.layers if isinstance(layer, Labels)]
        self.current_layer = next((layer for layer in labels_layers if layer.name == layer_name), None)
        
        if self.current_layer:
            # Check if layer is using single-color mode 
            # (color_mode is 'direct' or 'single', and color dict exists or is empty for single mode)
            color_mode = getattr(self.current_layer, 'color_mode', 'auto')
            color_dict = getattr(self.current_layer, 'color', {})
            is_single_color = (
                color_mode in ('direct', 'single') and
                (len(color_dict) > 0 or color_mode == 'single')
            )
            
            if is_single_color:
                self.single_color_check.setChecked(True)
                self.color_button.setEnabled(True)
                # Try to extract current color from the first label
                if self.current_layer.color:
                    first_color = next(iter(self.current_layer.color.values()))
                    if isinstance(first_color, (tuple, list, np.ndarray)) and len(first_color) >= 3:
                        # Convert from [0,1] range to [0,255] for QColor
                        r = int(first_color[0] * 255)
                        g = int(first_color[1] * 255)
                        b = int(first_color[2] * 255)
                        self._current_color = QColor(r, g, b)
                        hex_color = f"#{r:02X}{g:02X}{b:02X}"
                        self.color_label.setText(f"Current: {hex_color}")
                        self.color_label.setStyleSheet(f"padding: 5px; border: 1px solid gray; background-color: {hex_color};")
            else:
                self.single_color_check.setChecked(False)
                self.color_button.setEnabled(False)
                self.color_label.setText("Current: Multi-color")
                self.color_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
                
            # Update opacity slider
            if hasattr(self.current_layer, 'opacity'):
                self.opacity_slider.setValue(int(self.current_layer.opacity * 100))
                self.opacity_label.setText(f"{int(self.current_layer.opacity * 100)}%")
        else:
            self.color_button.setEnabled(False)
            
    def _on_color_mode_changed(self, state):
        """Handle color mode toggle."""
        enabled = state == Qt.Checked
        self.color_button.setEnabled(enabled)
        
        if self.current_layer and enabled:
            # Switch to single-color mode - use current button color or default
            if not hasattr(self, '_current_color'):
                self._current_color = QColor(255, 0, 0)  # Default red
            self._apply_single_color(self._current_color)
        elif self.current_layer and not enabled:
            # Switch back to multi-color mode - clear color dict and use auto mode
            self.current_layer.color = {}
            self.current_layer.color_mode = 'auto'
            # Trigger refresh by toggling visibility
            was_visible = self.current_layer.visible
            self.current_layer.visible = False
            self.current_layer.visible = was_visible
            self.color_label.setText("Current: Multi-color")
            self.color_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
            
    def _pick_color(self):
        """Open color picker dialog."""
        if not self.current_layer:
            return
            
        # Get current color if in single-color mode
        if hasattr(self, '_current_color'):
            color = QColorDialog.getColor(self._current_color, self, "Choose Layer Color")
        else:
            color = QColorDialog.getColor(QColor(255, 0, 0), self, "Choose Layer Color")
            
        if color.isValid():
            self._current_color = color
            self._apply_single_color(color)
            
    def _apply_single_color(self, qcolor):
        """Apply single color to current layer."""
        if not self.current_layer:
            return
            
        # Convert QColor to RGB tuple in [0, 1] range
        r = qcolor.red() / 255.0
        g = qcolor.green() / 255.0
        b = qcolor.blue() / 255.0
        # Use opacity from slider, not from color picker
        a = self.opacity_slider.value() / 100.0
        
        # Get unique label values that exist in the data
        if hasattr(self.current_layer, 'data') and self.current_layer.data.size > 0:
            unique_labels = np.unique(self.current_layer.data)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)
        else:
            unique_labels = np.array([], dtype=np.int32)
        
        if len(unique_labels) == 0:
            # No labels to color
            return
        
        # Try using DirectLabelColormap first (if available in this napari version)
        try:
            from napari.utils.colormaps import DirectLabelColormap
            
            # Create DirectLabelColormap with color dictionary
            color_tuple = (r, g, b, a)
            color_dict = {int(label): color_tuple for label in unique_labels}
            
            # Create the colormap
            colormap = DirectLabelColormap(color_dict=color_dict)
            
            # Set the colormap and color mode
            self.current_layer.colormap = colormap
            self.current_layer.color_mode = 'direct'
            
        except (ImportError, AttributeError, TypeError, ValueError):
            # Fallback: Use color property with dictionary
            color_tuple = (r, g, b, a)
            color_dict = {int(label): color_tuple for label in unique_labels}
            
            # Set color mode to 'direct' first
            self.current_layer.color_mode = 'direct'
            
            # Update colors by modifying the dict in place, then reassigning
            # This ensures napari's setter is called
            current_colors = getattr(self.current_layer, 'color', {})
            current_colors.clear()
            current_colors.update(color_dict)
            self.current_layer.color = current_colors
            
            # Trigger update via opacity change (subtle change triggers refresh)
            try:
                orig_opacity = self.current_layer.opacity
                self.current_layer.opacity = orig_opacity + 0.0001
                self.current_layer.opacity = orig_opacity
            except Exception:
                pass
        
        # Force display update by triggering data change
        try:
            # Small modification to trigger update
            data = self.current_layer.data
            # Create a view (doesn't copy data) and reassign
            self.current_layer.data = data.view()
        except Exception:
            try:
                # Fallback: just reassign
                self.current_layer.data = self.current_layer.data
            except Exception:
                pass
        
        # Update display
        hex_color = f"#{qcolor.red():02X}{qcolor.green():02X}{qcolor.blue():02X}"
        self.color_label.setText(f"Current: {hex_color}")
        self.color_label.setStyleSheet(f"padding: 5px; border: 1px solid gray; background-color: {hex_color};")
        
    def _on_opacity_changed(self, value):
        """Handle opacity slider change."""
        opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        
        if self.current_layer:
            self.current_layer.opacity = opacity
            # If in single-color mode, reapply color with new opacity
            if self.single_color_check.isChecked() and hasattr(self, '_current_color'):
                self._apply_single_color(self._current_color)
            
    def _shuffle_colors(self):
        """Shuffle colors for multi-color mode."""
        if self.current_layer:
            # Clear color dict and use auto mode (will use default cyclic colormap)
            self.current_layer.color = {}
            self.current_layer.color_mode = 'auto'
            # Trigger a refresh by toggling visibility
            was_visible = self.current_layer.visible
            self.current_layer.visible = False
            self.current_layer.visible = was_visible
            self.color_label.setText("Current: Multi-color (shuffled)")
            self.color_label.setStyleSheet("padding: 5px; border: 1px solid gray;")

