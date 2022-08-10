#!/bin/bash
set -e

# This is needed so that server.py does find TotalSegmentator installation. When calling server.py
# directly with docker run command, then somehow TotalSegmentator is not found.
# cd /app
# python server.py

# Using gunicorn  (timeout is in seconds)
cd /app
gunicorn --bind 0.0.0.0:5000 --timeout 1000 -w 1 server:app