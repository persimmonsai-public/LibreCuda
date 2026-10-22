#!/bin/sh
# Writes the symbol table of the libcuda.so.1 filter: an empty function for every driver API function ($1, the runtime
# library, exports). At run time the filter's symbols resolve to the runtime library's functions.
set -e
nm -D --defined-only "$1" | awk '$3 ~ /^cu[A-Z]/ { sub(/@.*/, "", $3); print $3 }' | sort -u |
    awk '{ print "extern \"C\" void " $1 "() {}" }' > "$2.tmp"
mv "$2.tmp" "$2"
