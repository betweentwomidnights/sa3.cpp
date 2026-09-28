import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
web_dir = os.path.join(root_dir, "web")
header_path = os.path.join(root_dir, "src", "embedded_train_web.h")

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

html = read_file(os.path.join(web_dir, "train.html"))
js = read_file(os.path.join(web_dir, "train.js"))

# Verify no content would break the raw string literal delimiter
for name, content in [("train.html", html), ("train.js", js)]:
    if ")sa3trainweb" in content:
        print(f"ERROR: {name} contains closing delimiter )sa3trainweb", file=sys.stderr)
        sys.exit(1)

def cpp_chunks(content):
    chunks = [content[i:i + 4000] for i in range(0, len(content), 4000)]
    return " +\n    ".join(f'std::string(R"sa3trainweb({chunk})sa3trainweb")' for chunk in chunks)

header = f"""#pragma once
// embedded_train_web.h — training web UI assets embedded in the binary at build time.
// AUTO-GENERATED from web/train.html and web/train.js. Do not edit by hand.
// Rebuild with: python3 tools/gen_embedded_train_web.py

#include <string>

namespace embedded_train_web {{

inline const std::string index_html =
    {cpp_chunks(html)};

inline const std::string train_js =
    {cpp_chunks(js)};

}} // namespace embedded_train_web
"""

if "--check" in sys.argv[1:]:
    try:
        with open(header_path, "r", encoding="utf-8") as f:
            current = f.read()
    except FileNotFoundError:
        current = None
    if current != header:
        print(f"{header_path} is stale; run python tools/gen_embedded_train_web.py", file=sys.stderr)
        sys.exit(1)
    print(f"Up to date: {header_path}")
else:
    with open(header_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
    print(f"Wrote {header_path}")
