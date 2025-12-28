with open("tui/screens/optimization_dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    start = max(0, 430)
    end = min(len(lines), 440)
    for i in range(start, end):
        print(f"{i+1}: {repr(lines[i])}")


