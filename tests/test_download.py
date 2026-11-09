import http.server
import threading
import time

import pytest
import requests

from totalsegmentator import libs


class _StalledWeightsHandler(http.server.BaseHTTPRequestHandler):
    # Sends the start of a zip file and then stops sending, as a stalled connection does.
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", "1000")
        self.end_headers()
        self.wfile.write(b"PK")
        self.wfile.flush()
        self.server.stalled.wait(10)

    def log_message(self, *args):
        pass


def test_download_url_and_unpack_fails_on_a_stalled_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(libs, "DOWNLOAD_TIMEOUT", 1)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "127.0.0.1")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StalledWeightsHandler)
    server.stalled = threading.Event()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        start = time.monotonic()
        with pytest.raises(requests.exceptions.ConnectionError, match="Read timed out"):
            libs.download_url_and_unpack(f"http://127.0.0.1:{server.server_port}/weights.zip", tmp_path)
        assert time.monotonic() - start < 5
    finally:
        server.stalled.set()
        server.shutdown()
        server.server_close()

    assert not (tmp_path / "tmp_download_file.zip").exists()
