#!/bin/bash
set -e

# This is needed so that server.py does find TotalSegmentator installation. When calling server.py
# directly with docker run command, then somehow TotalSegmentator is not found.
cd /app
python server.py