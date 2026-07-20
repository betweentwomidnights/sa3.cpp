import os
import sys

web_dir = "./web"
header_path = "./src/embedded_train_web.h"

def read_file(path):
    with open(path, "r") as f:
        return f.read()

html = read_file(os.path.join(web_dir, "train.html"))
js = read_file(os.path.join(web_dir, "train.js"))

# Verify no content would break the raw string literal delimiter
for name, content in [("train.html", html), ("train.js", js)]:
    if ")sa3trainweb" in content:
        print(f"ERROR: {name} contains closing delimiter )sa3trainweb", file=sys.stderr)
        sys.exit(1)

header = f"""#pragma once
// embedded_train_web.h — training web UI assets embedded in the binary at build time.
// AUTO-GENERATED from web/train.html and web/train.js. Do not edit by hand.
// Rebuild with: python3 tools/gen_embedded_train_web.py

namespace embedded_train_web {{

inline const char* index_html = R"sa3trainweb(
{html}
)sa3trainweb";

inline const char* train_js = R"sa3trainweb(
{js}
)sa3trainweb";

}} // namespace embedded_train_web
"""

with open(header_path, "w") as f:
    f.write(header)
print(f"Wrote {header_path}")
