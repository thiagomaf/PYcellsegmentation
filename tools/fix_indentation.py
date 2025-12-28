import os

filepath = "tui/screens/optimization_dashboard.py"
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if "except Exception as e:" in line:
        # Check if next line is the comment
        if i+1 < len(lines) and "# Don't raise - just log the error" in lines[i+1]:
            # Check if line after that is empty or dedented
            if i+2 < len(lines):
                next_line = lines[i+2]
                if not next_line.strip() or len(next_line) - len(next_line.lstrip()) <= len(line) - len(line.lstrip()):
                    # It's an empty block!
                    print(f"Fixing empty except block at line {i+1}")
                    # indent level
                    indent = line[:len(line) - len(line.lstrip())] + "    "
                    new_lines.append(f"{indent}import logging\n")
                    new_lines.append(f"{indent}logging.getLogger(__name__).error(f'Error: {{e}}')\n")

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)


