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
            # #region agent log
            try:
                import json, time
                widget_size = (self.size.width, self.size.height) if hasattr(self, 'size') else None
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H3,H4","location":"plotext_static.py:refresh_plot","message":"BEFORE build","data":{"widget_size":widget_size},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            
            # Build the plot as a string using plotext's build() method
            # This returns the plot as a string with ANSI color codes
            # IMPORTANT: plotext is a module-level singleton, so we need to ensure
            # the plot state is correct before building
            try:
                plot_text = self._plt.build()
            except Exception as build_error:
                # #region agent log
                try:
                    import json, time
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"BUILD_ERROR","location":"plotext_static.py:refresh_plot","message":"plt.build() failed","data":{"error":str(build_error),"error_type":type(build_error).__name__},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion
                self.update(f"Error building plot: {str(build_error)[:100]}")
                self.refresh()
                return
            # #region agent log
            try:
                import json, time
                plot_text_len = len(plot_text) if plot_text else 0
                plot_text_preview = plot_text[:100] if plot_text else None
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H3","location":"plotext_static.py:refresh_plot","message":"AFTER build","data":{"plot_text_len":plot_text_len,"plot_text_preview":plot_text_preview,"is_empty":not plot_text or plot_text.strip() == ""},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            
            # Check if plot_text is empty or None
            if not plot_text or plot_text.strip() == "":
                # Log why it's empty
                # #region agent log
                try:
                    import json, time
                    # Check what's in the plotext object
                    has_data = hasattr(self._plt, '_data') and len(self._plt._data) > 0 if hasattr(self._plt, '_data') else False
                    has_figure = hasattr(self._plt, '_figure') and len(self._plt._figure) > 0 if hasattr(self._plt, '_figure') else False
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"EMPTY","location":"plotext_static.py:refresh_plot","message":"Plot text is empty","data":{"has_data":has_data,"has_figure":has_figure,"plot_size":(self._plt.plotsize()[0], self._plt.plotsize()[1]) if hasattr(self._plt, 'plotsize') else None},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion
                self.update("Plot is empty - no data to display\nCheck that:\n- At least 2 parameters are enabled\n- Config files are added to the project")
                self.refresh()
                return
            
            # Mark that plot has been built
            self._plot_built = True
            
            # #region agent log
            try:
                import json, time
                plot_text_preview = plot_text[:200] if plot_text else None
                plot_text_lines = len(plot_text.split('\n')) if plot_text else 0
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"BEFORE_RICH","location":"plotext_static.py:refresh_plot","message":"About to convert to Rich Text","data":{"plot_text_len":len(plot_text),"plot_text_lines":plot_text_lines,"preview":plot_text_preview},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            
            # Use Rich Text to properly render ANSI codes
            # Rich's Text.from_ansi() will parse ANSI escape sequences into Rich markup
            try:
                rich_text = Text.from_ansi(plot_text)
                # #region agent log
                try:
                    import json, time
                    rich_text_len = len(str(rich_text)) if rich_text else 0
                    rich_text_repr = repr(rich_text)[:200] if rich_text else None
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H6,H7","location":"plotext_static.py:refresh_plot","message":"BEFORE update with Rich Text","data":{"rich_text_len":rich_text_len,"plot_text_len":len(plot_text),"rich_text_repr":rich_text_repr},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion
                
                # Try updating with the Rich Text object directly
                self.update(rich_text)
                self.refresh()
                
                # #region agent log - verify update worked
                try:
                    import json, time
                    # Check if widget actually has content now
                    widget_content = str(self.renderable) if hasattr(self, 'renderable') and self.renderable else None
                    widget_content_len = len(widget_content) if widget_content else 0
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"AFTER_UPDATE","location":"plotext_static.py:refresh_plot","message":"After Rich Text update","data":{"widget_content_len":widget_content_len,"has_content":widget_content_len > 0},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion
                
            except Exception as e:
                # If Rich Text update fails, try converting to string
                # #region agent log
                try:
                    import json, time
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H7","location":"plotext_static.py:refresh_plot","message":"Rich Text update failed, trying string","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion
                
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
                # #region agent log
                try:
                    import json, time
                    # Check what's actually in the widget after update
                    # Try multiple ways to access widget content
                    widget_content_renderable = None
                    widget_content_renderable_len = 0
                    if hasattr(self, 'renderable'):
                        try:
                            widget_content_renderable = str(self.renderable) if self.renderable else None
                            widget_content_renderable_len = len(widget_content_renderable) if widget_content_renderable else 0
                        except:
                            pass
                    # Try accessing _content if it exists
                    widget_content_private = None
                    widget_content_private_len = 0
                    if hasattr(self, '_content'):
                        try:
                            widget_content_private = str(self._content) if self._content else None
                            widget_content_private_len = len(widget_content_private) if widget_content_private else 0
                        except:
                            pass
                    # Try accessing content attribute directly (widget_attrs showed "content": "Text")
                    widget_content_direct = None
                    widget_content_direct_len = 0
                    if hasattr(self, 'content'):
                        try:
                            widget_content_direct = str(self.content) if self.content else None
                            widget_content_direct_len = len(widget_content_direct) if widget_content_direct else 0
                        except:
                            pass
                    # Try accessing render_str if it exists
                    widget_render_str = None
                    widget_render_str_len = 0
                    try:
                        if hasattr(self, 'render_str'):
                            widget_render_str = self.render_str()
                            widget_render_str_len = len(widget_render_str) if widget_render_str else 0
                    except Exception as e:
                        widget_render_str = f"ERROR: {e}"
                    # Also check if widget is visible and mounted
                    widget_visible = None
                    try:
                        widget_visible = self.is_visible if hasattr(self, 'is_visible') else None
                    except:
                        pass
                    widget_mounted = None
                    try:
                        widget_mounted = hasattr(self, '_node') and self._node is not None
                    except:
                        pass
                    # Check all attributes that might contain content
                    widget_attrs = {}
                    for attr in ['renderable', '_content', '_renderable', 'content']:
                        if hasattr(self, attr):
                            try:
                                val = getattr(self, attr)
                                widget_attrs[attr] = type(val).__name__ if val else None
                            except:
                                widget_attrs[attr] = "ERROR"
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H7,H8","location":"plotext_static.py:refresh_plot","message":"AFTER update with Rich Text and refresh","data":{"widget_content_renderable_len":widget_content_renderable_len,"widget_content_private_len":widget_content_private_len,"widget_content_direct_len":widget_content_direct_len,"widget_render_str_len":widget_render_str_len,"widget_visible":widget_visible,"widget_mounted":widget_mounted,"widget_attrs":widget_attrs,"widget_size":(self.size.width, self.size.height) if hasattr(self, 'size') else None},"timestamp":time.time()*1000})+"\n")
                except Exception as e:
                    try:
                        import json, time, traceback
                        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H8","location":"plotext_static.py:refresh_plot","message":"EXCEPTION checking widget content","data":{"error":str(e),"error_type":type(e).__name__,"traceback":traceback.format_exc()},"timestamp":time.time()*1000})+"\n")
                    except: pass
                # #endregion
            except Exception as e:
                # #region agent log
                try:
                    import json, time, traceback
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H6","location":"plotext_static.py:refresh_plot","message":"EXCEPTION in Rich Text parsing","data":{"error":str(e),"error_type":type(e).__name__,"traceback":traceback.format_exc()},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion
                # Fallback to plain text if Rich parsing fails
                # But try to at least remove visible ANSI codes
                import re
                # Remove ANSI escape sequences for fallback
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                clean_text = ansi_escape.sub('', plot_text)
                # #region agent log
                try:
                    import json, time
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H6","location":"plotext_static.py:refresh_plot","message":"Using fallback plain text","data":{"clean_text_len":len(clean_text),"has_content":bool(clean_text.strip())},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion
                if clean_text.strip():
                    self.update(clean_text)
                    # #region agent log
                    try:
                        import json, time
                        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H7","location":"plotext_static.py:refresh_plot","message":"AFTER update with clean text","data":{},"timestamp":time.time()*1000})+"\n")
                    except: pass
                    # #endregion
                else:
                    self.update("Plot rendered but appears empty")
        except Exception as e:
            import traceback
            error_msg = f"Error rendering plot: {e}\n{traceback.format_exc()[:200]}"
            self.update(error_msg)

