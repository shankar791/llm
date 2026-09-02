import http.server
import os
import sys
import mimetypes
import re

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.normpath(os.path.join(DIRECTORY, '..', 'satquery-frontend-dashboard'))

# Ensure common 3D and media extensions have proper MIME types
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".basis")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("audio/ogg", ".ogg")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        """Serve a GET request with Range support and dashboard routing."""
        # Route /mission and /monitor to the dashboard
        url_path = self.path.split('?')[0].split('#')[0]  # strip query/fragment
        if url_path == '/mission' or url_path == '/monitor':
            # Serve the dashboard index.html
            dashboard_index = os.path.join(DASHBOARD_DIR, 'index.html')
            if os.path.exists(dashboard_index):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                with open(dashboard_index, 'rb') as f:
                    content = f.read()
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "Dashboard not found")
                return

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().do_GET()

        if not os.path.exists(path):
            return super().do_GET()

        range_header = self.headers.get("Range")
        if not range_header:
            return super().do_GET()

        try:
            total_size = os.path.getsize(path)
            range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if not range_match:
                return super().do_GET()

            start_str, end_str = range_match.groups()
            start = int(start_str)
            end = int(end_str) if end_str else total_size - 1

            if start >= total_size:
                self.send_error(416, "Requested Range Not Satisfiable")
                return

            end = min(end, total_size - 1)
            if start > end:
                self.send_error(416, "Requested Range Not Satisfiable")
                return

            length = end - start + 1
            content_type, _ = mimetypes.guess_type(path)
            if not content_type:
                content_type = "application/octet-stream"

            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{total_size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()

            with open(path, "rb") as f:
                f.seek(start)
                bytes_to_send = length
                chunk_size = 64 * 1024
                while bytes_to_send > 0:
                    read_len = min(chunk_size, bytes_to_send)
                    data = f.read(read_len)
                    if not data:
                        break
                    self.wfile.write(data)
                    bytes_to_send -= len(data)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception:
            pass

def run():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    server_address = ("", port)
    
    with http.server.ThreadingHTTPServer(server_address, RangeHTTPRequestHandler) as httpd:
        url = f"http://localhost:{port}/"
        print(f"==================================================")
        print(f"  EDOLUS Multithreaded 3D Web Server Running!")
        print(f"  Local URL: {url}")
        print(f"  Serving Directory: {DIRECTORY}")
        print(f"  Press Ctrl+C to stop.")
        print(f"==================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    run()
