import re

content = open('alkame_dashboard.html', encoding='utf-8').read()
handlers = re.findall(r'on[a-z]+\s*=\s*["\']([^"\']+)["\']', content)
print(f"Total inline event handlers found: {len(handlers)}")
funcs = set()
for h in handlers:
    m = re.match(r'([a-zA-Z0-9_]+)\s*\(', h.strip())
    if m:
        funcs.add(m.group(1))

print("\n=== FUNCTION AUDIT ===")
missing = []
for f in sorted(funcs):
    exists = f"function {f}" in content
    status = "EXISTS" if exists else "MISSING!"
    if not exists:
        missing.append(f)
    print(f"  {f}: {status}")

print(f"\nMissing functions: {missing}")

# Check getElementById calls
dom_ids = re.findall(r'getElementById\(["\']([^"\']+)["\']\)', content)
print(f"\nTotal getElementById calls: {len(dom_ids)}")
missing_ids = []
for did in sorted(set(dom_ids)):
    pattern = f'id="{did}"'
    pattern_single = f"id='{did}'"
    if pattern not in content and pattern_single not in content:
        missing_ids.append(did)

print(f"Missing DOM IDs called by JS: {missing_ids}")
