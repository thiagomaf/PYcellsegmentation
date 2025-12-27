# PyCellSegmentation TUI Manual

The PyCellSegmentation TUI (Terminal User Interface) provides a powerful, console-based environment for managing segmentation projects, configuring Cellpose parameters, and monitoring execution pipelines.

## Table of Contents

1. [Installation](#installation)
2. [Running the TUI](#running-the-tui)
3. [Interface Overview](#interface-overview)
4. [Workflows](#workflows)
   - [Creating a New Project](#creating-a-new-project)
   - [Managing Images](#managing-images)
   - [Configuring Parameters](#configuring-parameters)
   - [Monitoring the Pipeline](#monitoring-the-pipeline)
   - [Exploring Results](#exploring-results)
5. [Keyboard Shortcuts](#keyboard-shortcuts)
6. [Troubleshooting](#troubleshooting)

## Installation

From the project root directory, install the required dependencies for the TUI:

```bash
pip install -r tui/requirements.txt
```

For image preview capabilities in the Result Explorer, install the optional dependency:
```bash
pip install textual-imageview
```

## Running the TUI

To start the application, run the following command from the project root:

```bash
python -m tui.app
```

### Command-Line Arguments

The TUI supports command-line arguments to load a project file and optionally open a specific view:

**Load a Project File:**
```bash
python -m tui.app --project path/to/project.opt.json
# or using the short form:
python -m tui.app -p path/to/project.opt.json
```

**Load a Project and Open a Specific View:**
```bash
# Open the parameter editor directly
python -m tui.app --project path/to/project.opt.json --view parameters
# or using the convenience flag:
python -m tui.app --project path/to/project.opt.json --parameters

# Open the optimization dashboard
python -m tui.app --project path/to/project.opt.json --view optimization
```

**Available View Options:**
- `--view parameters` or `--view params` - Opens the parameter ranges editor
- `--view optimization` or `--view opt` - Opens the optimization dashboard
- `--parameters` - Shortcut flag equivalent to `--view parameters`
- `--optimization` - Shortcut flag equivalent to `--view optimization`

**Examples:**
```bash
# Load project and open parameter editor
python -m tui.app -p my_project.opt.json --parameters

# Load project and open optimization dashboard
python -m tui.app --project my_project.opt.json --view opt
# or using the convenience flag:
python -m tui.app --project my_project.opt.json --optimization
```

**Developer Mode:**
If you need to debug layout issues or inspect widgets, run in developer mode (requires `textual-dev`):

```bash
textual run --dev tui/app.py
```
*Press `F12` or `Ctrl+I` in this mode to open the widget inspector.*

**Note:** When using command-line arguments with developer mode, pass them after the script path:
```bash
textual run --dev tui/app.py --project my_project.opt.json --parameters
```

## Interface Overview

The TUI is divided into two main sections: the **Dashboard** and the **Project Editor**.

### 1. Dashboard
The entry point of the application.
- **New Project**: Start a configuration from scratch.
- **Load Project**: Open an existing JSON configuration file.
- **Exit**: Quit the application.

### 2. Project Editor
The main workspace, accessible after creating or loading a project. It has a sidebar with three main views:

#### A. Config Editor (`c`)
The core configuration area, split into sub-tabs:
- **General (`1`)**: Basic project settings (Name, Root Directory, etc.).
- **Images (`2`)**: Manage the list of images to process.
  - *Features*: Add/Remove images, inline editing, copy/paste settings.
- **Parameters (`3`)**: Configure Cellpose segmentation parameters.
- **Preview (`4`)**: (Experimental) Preview segmentation settings.

#### B. Pipeline Status (`p`)
Monitors the execution of the segmentation pipeline.
- **Live Status**: Shows if the pipeline is Running, Idle, or Stale.
- **Steps**: specific processing steps (e.g., Preprocessing, Segmentation).
- **Logs**: Real-time log output from the processing engine.
- **Progress**: Visual progress bar.

#### C. Result Explorer (`r`)
Browse and inspect output files.
- **Tree View**: Hierarchical view of images and their result sets.
- **Metadata**: Detailed information about selected files.
- **Preview**: View content of JSON, CSV, and Image files (masks/TIFFs).

## Workflows

### Creating a New Project
1. Select **New Project** from the Dashboard.
2. You will be taken to the **Config Editor** > **General** tab.
3. Enter a **Project Name** and **Project Root Directory**.
4. Press `Ctrl+S` to save the project configuration to a JSON file.

### Managing Images
1. Navigate to **Config Editor** > **Images** (`2`).
2. **Add Image**: Click the "Add Image" button to append a new entry.
3. **Edit**: 
   - **Double-click** a cell to edit values inline.
   - **Spacebar** toggles boolean (checkbox) values like `Active` or `Segmentation`.
   - **Enter** opens a detailed editor for the selected cell.
4. **Bulk Actions**:
   - `Ctrl+C` to copy a cell's value.
   - `Ctrl+V` to paste the value into other cells.

### Configuring Parameters
1. Navigate to **Config Editor** > **Parameters** (`3`).
2. Create or edit parameter sets (e.g., `cyto2`, `nuclei`).
3. Adjust settings like `diameter`, `flow_threshold`, and `cellprob_threshold`.

### Monitoring the Pipeline
1. Once the pipeline script is running (externally), switch to **Pipeline Status** (`p`).
2. The indicator at the top left shows the system heartbeat:
   - 🟢 **Green**: Live and Running.
   - 🟠 **Orange**: Running but Stale (no recent updates).
   - 🔴 **Red**: Stopped or Error.
   - 🔵 **Blue**: Completed.

### Exploring Results
1. Switch to **Result Explorer** (`r`).
2. Expand the tree to see Images -> Parameter Sets -> Output Files.
3. Select a file to view its details.
   - **Masks (.tif)**: Shows a colorized preview of the segmentation mask (if `textual-imageview` is installed) or a text-based approximation.
   - **Data (.json/csv)**: Shows syntax-highlighted content.

## Keyboard Shortcuts

| Context | Shortcut | Action |
|---------|----------|--------|
| **Global** | `Ctrl+Q` | Quit Application |
| | `F12` / `Ctrl+I` | Open Inspector (Dev Mode) |
| **Editor** | `Ctrl+S` | Save Project |
| | `Esc` | Back / Cancel |
| | `c` | Switch to Config Editor |
| | `p` | Switch to Pipeline Status |
| | `r` | Switch to Result Explorer |
| **Config** | `1` | View General Settings |
| | `2` | View Images |
| | `3` | View Parameters |
| | `4` | View Preview |
| **Tables** | `Space` | Toggle Checkbox |
| | `Enter` | Edit Cell |
| | `Ctrl+C` | Copy Cell |
| | `Ctrl+V` | Paste Cell |

## Troubleshooting

### Layout Issues
If the interface looks broken or text is overlapping:
1. Resize your terminal window.
2. Ensure you are using a strictly monospaced font (e.g., Fira Code, Consolas).
3. Run in developer mode (`textual run --dev ...`) to inspect elements.

### Pipeline Not Updating
- Check if the **Project Root** in "General" settings matches the directory where the pipeline script is running.
- Ensure the pipeline script has write permissions to the status file.
- Look for the "Last Updated" timestamp in the Pipeline Status footer.

### Images Not Loading
- Verify the `original_image_filename` paths are correct relative to the project root or are absolute paths.
- Check the logs in **Pipeline Status** for "File not found" errors.
