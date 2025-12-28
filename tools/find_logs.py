with open("tui/screens/optimization_dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "# #region agent log" in line:
            print(f"{i+1}: {repr(line)}")


