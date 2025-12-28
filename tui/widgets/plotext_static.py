"""Custom widget that uses plotext directly to render plots as text."""
from textual.widgets import Static
from rich.text import Text
from typing import Optional
import plotext as plt

class PlotextStatic(Static):
    """A Static widget that displays plotext plots by rendering them as text.
    
    This widget uses plotext directly (not through textual-plotext wrapper)
    to have full control over colors and other features.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the widget."""
        super().__init__("Initializing plot...", *args, **kwargs)
        # IMPORTANT: plotext is a module-level singleton, so all widgets share the same instance
        # We need to store plot configuration and rebuild it in refresh_plot()
        import plotext as plt_module
        self._plt = plt_module
        self._plot_config = None  # Store plot configuration to rebuild on refresh
    
    @property
    def plt(self):
        """Access to the underlying plotext object."""
        return self._plt
    
    def on_mount(self) -> None:
        """Called when widget is mounted - set initial plot size based on widget size."""
        self._update_plot_size()
        # Test: Try to show a simple test plot to verify widget is working
        # This will be overwritten when update_visualization is called
        try:
            self._plt.clear_data()
            self._plt.clear_figure()
            self._plt.title("Loading plot...")
            self._plt.xlabel("")
            self._plt.ylabel("")
            # Build and show immediately to verify widget works
            test_plot = self._plt.build()
            if test_plot and test_plot.strip():
                from rich.text import Text
                try:
                    self.update(Text.from_ansi(test_plot))
                except:
                    self.update(test_plot)
                self.refresh()
        except Exception:
            # If test plot fails, just keep the initializing message
            pass
    
    def on_resize(self) -> None:
        """Called when widget is resized - update plot size and refresh."""
        self._update_plot_size()
        # If plot has been built before, refresh it with new size
        try:
            if hasattr(self, '_plot_built') and self._plot_built:
                self.refresh_plot()
        except Exception:
            pass
    
    def _update_plot_size(self) -> None:
        """Update plotext plot size to match widget dimensions."""
        try:
            # Get widget size - use the full available space
            size = self.size
            if size.width > 0 and size.height > 0:
                # Set plot size to match widget exactly
                # The widget size already accounts for its container (borders, padding)
                # so we can use the full size. Plotext will handle its own internal spacing
                width = max(40, size.width)  # Use full width
                height = max(15, size.height)  # Use full height
                self._plt.plotsize(int(width), int(height))
        except Exception:
            # If size not available yet, use defaults
            self._plt.plotsize(60, 25)
    
    def refresh_plot(self) -> None:
        """Refresh the plot by building it and updating the widget."""
        try:
            # CRITICAL FIX: Rebuild plot from stored configuration to avoid shared state issues
            if hasattr(self, '_plot_config') and self._plot_config:
                config = self._plot_config
                self._plt.clear_figure()
                self._plt.theme("dark")
                self._plt.title(config['title'])
                self._plt.xlabel(config['xlabel'])
                self._plt.ylabel(config['ylabel'])
                
                # Plot confidence intervals if available
                if config.get('y_std') and len(config['y_std']) > 0:
                    try:
                        y_mean = config['y_mean']
                        y_std = config['y_std']
                        if len(y_std) == len(y_mean) == len(config['x_pred']):
                            y_upper = [(m + s) for m, s in zip(y_mean, y_std)]
                            y_lower = [(m - s) for m, s in zip(y_mean, y_std)]
                            self._plt.plot(config['x_pred'], y_upper, color=(38, 38, 38), label="")
                            self._plt.plot(config['x_pred'], y_lower, color=(38, 38, 38), label="")
                    except Exception:
                        pass
                
                # Plot main data
                if len(config['x_pred']) == len(config['acquisition']):
                    self._plt.plot(config['x_pred'], config['acquisition'], color="blue", label="A(x)")
                if len(config['x_pred']) == len(config['y_mean']):
                    self._plt.plot(config['x_pred'], config['y_mean'], color="cyan", label="μ(x)")
                if len(config['x_obs']) == len(config['y_obs']) and len(config['x_obs']) > 0:
                    self._plt.scatter(config['x_obs'], config['y_obs'], color="red", label="Observed")
            
            # Check if widget is mounted - use multiple checks
            is_mounted = (
                hasattr(self, '_node') and self._node is not None
            ) or (
                hasattr(self, 'app') and self.app and hasattr(self.app, 'screen') and self.app.screen
            )
            
            # If not mounted, try to schedule update, but also try to update anyway
            # (widget might be mounted but _node not set yet)
            if not is_mounted:
                # Try to schedule update after refresh
                if hasattr(self, 'app') and self.app:
                    try:
                        self.app.call_after_refresh(self.refresh_plot)
                    except:
                        pass
                # Don't return early - try to update anyway in case widget is ready
            
            # Update plot size before building (use current widget size)
            # Only update if widget has valid size
            if hasattr(self, 'size') and self.size.width > 0 and self.size.height > 0:
                self._update_plot_size()
            else:
                # Set a default size for plotext
                self._plt.plotsize(70, 30)
            
            # Build the plot as a string using plotext's build() method
            # This returns the plot as a string with ANSI color codes
            # IMPORTANT: plotext is a module-level singleton, so we need to ensure
            # the plot state is correct before building
            try:
                plot_text = self._plt.build()
            except Exception as build_error:
                self.update(f"Error building plot: {str(build_error)[:100]}")
                self.refresh()
                return
            
            # Check if plot_text is empty or None
            if not plot_text or plot_text.strip() == "":
                # Log why it's empty
                self.update("Plot is empty - no data to display\nCheck that:\n- At least 2 parameters are enabled\n- Config files are added to the project")
                self.refresh()
                return
            
            # Mark that plot has been built
            self._plot_built = True
            
            
            # Use Rich Text to properly render ANSI codes
            # Rich's Text.from_ansi() will parse ANSI escape sequences into Rich markup
            try:
                rich_text = Text.from_ansi(plot_text)
                
                # Try updating with the Rich Text object directly
                self.update(rich_text)
                self.refresh()
                
                
            except Exception as e:
                # If Rich Text update fails, try converting to string
                
                # Fallback: try updating with raw plot_text as string (Textual should handle ANSI)
                try:
                    self.update(plot_text)
                    self.refresh()
                except Exception as e2:
                    # Last resort: strip ANSI and use plain text
                    import re
                    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                    clean_text = ansi_escape.sub('', plot_text)
                    if clean_text.strip():
                        self.update(clean_text)
                        self.refresh()
                    else:
                        # Last resort: show error message
                        error_msg = f"Error displaying plot:\n{str(e)[:50]}\n{str(e2)[:50]}\n\nPlot text length: {len(plot_text) if plot_text else 0}"
                        self.update(error_msg)
                        self.refresh()
            
            # Final check: if widget is still empty after all attempts, show a message
            try:
                # Check if widget has any content
                if hasattr(self, 'renderable'):
                    content = str(self.renderable) if self.renderable else ""
                    if not content or len(content.strip()) < 10:
                        # Widget is empty, show fallback message
                        self.update("Plot widget is ready but no content to display.\nThis may indicate:\n- No data points\n- Plot setup incomplete\n- Display error")
                        self.refresh()
            except Exception:
                # If we can't check, at least try to show something
                try:
                    self.update("Plot widget initialized")
                    self.refresh()
                except:
                    pass
            except Exception as e:
                # Fallback to plain text if Rich parsing fails
                # But try to at least remove visible ANSI codes
                import re
                # Remove ANSI escape sequences for fallback
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                clean_text = ansi_escape.sub('', plot_text)
                if clean_text.strip():
                    self.update(clean_text)
                else:
                    self.update("Plot rendered but appears empty")
        except Exception as e:
            import traceback
            error_msg = f"Error rendering plot: {e}\n{traceback.format_exc()[:200]}"
            self.update(error_msg)

