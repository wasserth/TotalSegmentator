#!/usr/bin/env python3
"""
Start local HTTP server for TotalSegmentator web viewer
Server root = project root (portable across machines)
"""

import http.server
import socketserver
from pathlib import Path
import os
import sys
import webbrowser
import time
import threading

# -------------------------------------------------
# Resolve project root
# totalsegmentator/bin/start_viewer_server.py
# -> project_root = parents[2]
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_VIEWER_DIR = PROJECT_ROOT / "totalsegmentator" / "bin" / "web_viewer"

if not WEB_VIEWER_DIR.exists():
    print(f"❌ web_viewer not found at {WEB_VIEWER_DIR}")
    sys.exit(1)

# -------------------------------------------------
# Find available STL directories
# -------------------------------------------------
def find_stl_directories():
    """Find all directories containing stl_list.json"""
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        return []
    
    stl_dirs = []
    for json_file in data_dir.rglob("stl_list.json"):
        stl_dir = json_file.parent
        # Get relative path from project root
        rel_path = stl_dir.relative_to(PROJECT_ROOT)
        stl_dirs.append("/" + str(rel_path))

    
    return sorted(stl_dirs)

# -------------------------------------------------
# Start HTTP server from PROJECT_ROOT
# -------------------------------------------------
os.chdir(PROJECT_ROOT)

PORT = 8000

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

print("=" * 70)
print("🌐 TotalSegmentator Enhanced Web Viewer Server")
print("=" * 70)
print(f"📁 Server root: {PROJECT_ROOT}")
print(f"📄 Viewer:      viewer_enhanced.html (NEW)")
print(f"🚀 URL:         http://localhost:{PORT}")
print("=" * 70)

# Find STL directories
stl_dirs = find_stl_directories()

if stl_dirs:
    print(f"\n✅ Found {len(stl_dirs)} STL directory(ies):")
    for i, dir_path in enumerate(stl_dirs, 1):
        url = f"http://localhost:{PORT}/totalsegmentator/bin/web_viewer/viewer_enhanced.html?dir={dir_path}"
        print(f"\n{i}. {dir_path}")
        print(f"   URL: {url}")
    
    # Auto-open first one
    first_url = f"http://localhost:{PORT}/totalsegmentator/bin/web_viewer/viewer_enhanced.html?dir={stl_dirs[0]}"
    
    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        print(f"\n🌐 Opening browser: {first_url}")
        webbrowser.open(first_url)
    
    threading.Thread(target=open_browser, daemon=True).start()
else:
    print("\n⚠️  No STL directories found!")
    print("   Please run totalseg_gui.py first to generate STL files.")
    print("\n💡 Expected structure:")
    print("   data/2/output/out_total_all/total_all/stl_list.json")

print("\n" + "=" * 70)
print("🔥 Server is running. Press Ctrl+C to stop.")
print("=" * 70 + "\n")

with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        sys.exit(0)