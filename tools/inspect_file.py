with open("tui/screens/optimization_dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    # Check around line 408
    for i in range(405, 415):
        if i < len(lines):
            print(f"{i+1}: {repr(lines[i])}")


