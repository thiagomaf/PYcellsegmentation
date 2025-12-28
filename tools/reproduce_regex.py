import re

content = """
            scan_done = time.time()
            # #region agent log
            try:
                with open(r"g:\\My Drive\\Github\\PYcellsegmentation\\.cursor\\debug.log", "a", encoding="utf-8") as f:
                    import json
                    f.write("test")
            except: pass
            # #endregion
            
            # Check if new parameter sets were added
"""

pattern = r"^[ \t]*# #region agent log[\s\S]*?# #endregion(\r?\n)?"
matches = list(re.finditer(pattern, content, flags=re.MULTILINE))
print(f"Found {len(matches)} matches")
if matches:
    print(f"Match length: {len(matches[0].group())}")


