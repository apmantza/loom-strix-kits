#!/bin/bash
# Download this kit's pinned files (pins.json) and verify every sha256.
exec python3 "$(dirname "$0")/../../common/fetch.py" inkling-2-mini "$@"
