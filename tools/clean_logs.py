import os
import re

def clean_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping {filepath} due to encoding issue")
        return
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    # Regex to find the blocks
    pattern = r"^[ \t]*# #region agent log[\s\S]*?# #endregion(\r?\n)?"
    
    matches = list(re.finditer(pattern, content, flags=re.MULTILINE))
    if matches:
        print(f"Found {len(matches)} matches in {filepath}")
        new_content = re.sub(pattern, "", content, flags=re.MULTILINE)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

def main():
    root_dir = "."
    print(f"Scanning {os.path.abspath(root_dir)}")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter out hidden directories (starting with .) but allow current directory (.)
        parts = dirpath.split(os.sep)
        if any(part.startswith('.') and part != '.' for part in parts):
            continue
            
        if "venv" in dirpath or "__pycache__" in dirpath:
            continue
            
        for filename in filenames:
            if filename.endswith(".py"):
                clean_file(os.path.join(dirpath, filename))

if __name__ == "__main__":
    main()
