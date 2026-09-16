import sys

file_path = "ensemble_manager.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

if "import hashlib" not in content[:500]:
    content = "import hashlib\n" + content

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
