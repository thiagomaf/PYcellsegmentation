# PyCellSegmentation TUI

A modern Terminal User Interface (TUI) for managing PyCellSegmentation project configurations.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the TUI

From the project root directory:

```bash
python -m tui.app
```

Or:

```bash
cd tui
python app.py
```

## Features

- **Create New Projects**: Start a new configuration from scratch
- **Load Existing Projects**: Open and edit existing configuration files
- **Image Management**: Add, edit, and manage image configurations
- **Parameter Management**: Configure Cellpose segmentation parameters
- **Modern UI**: Beautiful, intuitive interface built with Textual

## Keyboard Shortcuts

- `Ctrl+Q` or `Esc`: Exit the application
- `Ctrl+S`: Save current project
- `Tab`: Navigate between fields
- `Enter`: Confirm/Submit
- Arrow keys: Navigate lists and tables

## Debugging Layout Issues

To debug TUI layout (similar to browser DevTools):

1. **Install developer tools** (if not already installed):
   ```bash
   pip install textual-dev
   ```

2. **IMPORTANT: Run with developer mode enabled**:
   ```bash
   textual run --dev tui/app.py
   ```
   Or from project root:
   ```bash
   textual run --dev -m tui.app
   ```
   
   **Note**: You MUST use `textual run --dev`. Running with `python -m tui.app` directly will NOT enable the inspector, even if textual-dev is installed.

3. **Open the Inspector**:
   - Press `F12` or `Ctrl+I` while the app is running
   - This shows the widget tree, CSS styles, and widget properties
   - Navigate with arrow keys, select widgets with Enter

See `tui/DEBUGGING.md` for more detailed debugging information.
