#!/bin/sh
# Writes weak definitions of every function the public header $1 declares: they return CUBLAS_STATUS_NOT_SUPPORTED (15)
# unless the library defines the function itself (its definition wins at link time). Programs calling an unimplemented
# function get an error status instead of failing to load or aborting on an unresolved symbol.
set -e
# (the name may follow on the next line)
tr '\n' ' ' < "$1" | grep -ohE 'CUBLASWINAPI +[A-Za-z0-9_]+ *\(' | sed -E 's/CUBLASWINAPI +//; s/ *\(//' | sort -u |
    awk '{ print "extern \"C\" __attribute__((weak)) int " $1 "() { return 15; }" }' > "$2.tmp"
mv "$2.tmp" "$2"
