import os
import re

def clean_file(filepath):
    print(f"Checking {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    # Regex to find the blocks
    pattern = r"^[ \t]*# #region agent log[\s\S]*?# #endregion(\r?\n)?"
    
    matches = list(re.finditer(pattern, content, flags=re.MULTILINE))
    if matches:
        print(f"Found {len(matches)} matches in {filepath}")
        new_content = re.sub(pattern, "", content, flags=re.MULTILINE)
        
        # Verify replacement
        if len(new_content) == len(content):
             print("Warning: Content length unchanged despite matches found!")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("Write successful")
        except Exception as e:
             print(f"Error writing {filepath}: {e}")

def main():
    filepath = "tui/screens/optimization_dashboard.py"
    clean_file(filepath)

if __name__ == "__main__":
    main()


