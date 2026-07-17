import os, sys

web_dir = "./web"
header_path = "./src/embedded_web.h"

def read_file(path):
    with open(path, "r") as f:
        return f.read()

html = read_file(os.path.join(web_dir, "index.html"))
js = read_file(os.path.join(web_dir, "app.js"))

# Verify no content would break the raw string literal delimiter
for name, content in [("index.html", html), ("app.js", js)]:
    if ")sa3web" in content:
        print(f"ERROR: {name} contains closing delimiter )sa3web", file=sys.stderr)
        sys.exit(1)

header = f"""#pragma once
// embedded_web.h — web UI assets embedded in the binary at build time.
// AUTO-GENERATED from web/index.html and web/app.js. Do not edit by hand.
// Rebuild with: python3 tools/gen_embedded_web.py

namespace embedded_web {{

inline const char* index_html = R"sa3web(
{html}
)sa3web";

inline const char* app_js = R"sa3web(
{js}
)sa3web";

}} // namespace embedded_web
"""

with open(header_path, "w") as f:
    f.write(header)
print(f"Wrote {header_path}")
