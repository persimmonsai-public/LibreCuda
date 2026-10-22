#!/bin/sh
# Writes weak definitions of every function the public header $1 declares (with the calling convention macro $3,
# default CUBLASWINAPI): they return the status $4 (default 15, CUBLAS_STATUS_NOT_SUPPORTED) unless the library defines
# the function itself (its definition wins at link time). Programs calling an unimplemented function get an error
# status instead of failing to load or aborting on an unresolved symbol.
set -e
macro="${3:-CUBLASWINAPI}"
status="${4:-15}"
# (the name may follow on the next line)
tr '\n' ' ' < "$1" | grep -ohE "$macro +[A-Za-z0-9_]+ *\\(" | sed -E "s/$macro +//; s/ *\\(//" | grep -v '^__' | sort -u |
    awk -v status="$status" '{ print "extern \"C\" __attribute__((weak)) int " $1 "() { return " status "; }" }' \
    > "$2.tmp"
mv "$2.tmp" "$2"
