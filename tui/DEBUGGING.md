# Debugging Textual TUI Layout

This guide explains how to debug your TUI layout, similar to using browser developer tools (F11 + inspect HTML/CSS).

## Installation

First, install the Textual developer tools:

```bash
pip install textual-dev
```

Or add it to your requirements:
```
textual-dev>=0.60.0
```

## Using the Inspector (Like Browser DevTools)

The **Inspector** is Textual's equivalent of browser developer tools. It shows you:
- The widget tree (like the DOM tree)
- CSS styles applied to each widget
- Widget properties and dimensions
- Computed styles

### Opening the Inspector

**IMPORTANT**: The Inspector **ONLY works when you run your app with `textual run --dev`**. 
Even if `textual-dev` is installed, running with `python -m tui.app` directly will NOT enable the inspector.

**Method 1: Run with Developer Mode (REQUIRED)**
You MUST run your app using the `textual` command with the `--dev` flag:
```bash
textual run --dev tui/app.py
```

Or from the project root:
```bash
textual run --dev -m tui.app
```

**Method 2: Keyboard Shortcut**
- After starting with `textual run --dev`, press `F12` or `Ctrl+I` while the app is running
- This opens the inspector overlay

**Why this is required**: The `textual run --dev` command enables developer mode, which:
- Loads the Inspector screen from textual-dev
- Enables hot-reloading of CSS
- Provides additional debugging features
- Makes the inspector action available

**If you see an error**: The error message will tell you to run with `textual run --dev`. This is the correct way to enable the inspector.

**Troubleshooting: If Inspector Still Doesn't Work**

1. **Verify your setup**:
   ```bash
   python tui/verify_setup.py
   ```
   This will check if textual-dev is installed and provide guidance.

2. **Make sure you're using the right command**:
   - ✅ **Correct**: `textual run --dev tui/app.py`
   - ❌ **Wrong**: `python -m tui.app` or `python tui/app.py`
   
   The inspector **only works** with the `textual run --dev` command.

2. **Verify textual-dev installation**:
   ```bash
   # Activate your venv first
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   
   pip show textual-dev
   ```

3. **Try running with explicit dev mode**:
   ```bash
   textual run --dev tui/app.py
   ```

4. **Check for error messages**: When you press F12 or Ctrl+I, you should see a notification if something is wrong. The app will show a warning message if textual-dev is not found.

### Navigating the Inspector

Once open, you can:
- **Navigate the widget tree**: Use arrow keys to move up/down
- **Select widgets**: Press Enter to select a widget and see its details
- **View styles**: See all CSS rules applied to the selected widget
- **View properties**: See widget dimensions, classes, IDs, etc.
- **Exit**: Press `Esc` or `Ctrl+I` again to close

## Other Debugging Methods

### 1. Print Widget Tree

Add this to your code to print the widget tree:

```python
def on_mount(self) -> None:
    """Print widget tree for debugging."""
    print(self.app.dom_tree)
```

### 2. Query Widgets

Use Textual's query system to inspect widgets:

```python
# Find a widget by ID
widget = self.query_one("#my-widget-id")

# Find all widgets with a class
widgets = self.query(".my-class")

# Print widget information
print(f"Widget: {widget}")
print(f"Size: {widget.size}")
print(f"Region: {widget.region}")
print(f"Classes: {widget.classes}")
```

### 3. Log Widget Events

Enable debug logging to see widget events:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 4. Screenshot/Dump

Textual can export the current screen state:

```python
# In your app code
self.export_screenshot(path="screenshot.png")
```

Or use the CLI:
```bash
textual screenshot your_app.py
```

### 5. CSS Debugging

To debug CSS issues:

1. **Check CSS loading**: Verify your CSS file is being loaded
   ```python
   CSS_PATH = str(Path(__file__).parent / "styles.tcss")
   ```

2. **View computed styles**: Use the Inspector (Ctrl+I) to see which styles are applied

3. **Test CSS rules**: Temporarily add borders to see widget boundaries:
   ```css
   * {
       border: solid red;
   }
   ```

4. **Check CSS specificity**: More specific selectors override general ones
   - Type selectors: `Button { }`
   - Class selectors: `.my-class { }`
   - ID selectors: `#my-id { }`
   - Pseudo-classes: `Button:hover { }`

## Common Layout Issues

### Widgets Not Visible
- Check `display: none` in CSS
- Check widget is actually mounted
- Check `opacity: 0` or `visibility: hidden`

### Layout Not Working
- Check container layout type: `layout: horizontal` or `layout: vertical`
- Check width/height constraints
- Check `dock` properties (top, bottom, left, right)

### Styles Not Applying
- Check CSS file is loaded
- Check selector specificity
- Check if styles are scoped (Textual 0.38+)
- Use Inspector to see which styles are actually applied

## Quick Reference

| Browser DevTools | Textual Equivalent |
|-----------------|-------------------|
| F11 / Inspect Element | Ctrl+I (Inspector) |
| Elements Tab | Widget Tree in Inspector |
| Styles Panel | Styles Panel in Inspector |
| Console | Python logging / print() |
| Network Tab | N/A (terminal app) |

## Example: Debugging a Layout Issue

1. **Run your app**
2. **Press Ctrl+I** to open inspector
3. **Navigate to the problematic widget** using arrow keys
4. **Press Enter** to select it
5. **Check the Styles panel** to see:
   - Which CSS rules are applied
   - Computed dimensions
   - Layout properties
6. **Modify your CSS** based on what you see
7. **Reload the app** to see changes

## Tips

- The Inspector shows the **live widget tree** - changes as you interact with the app
- You can inspect widgets even in modal dialogs
- Use `textual run --dev` for additional debugging features
- Check Textual documentation for more advanced debugging: https://textual.textualize.io/

