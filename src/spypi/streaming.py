import io
import logging
from http import server
from threading import Condition, Lock
import socketserver

# Global output lock and latest frame variable.
output_lock = Lock()
_latest_frame = None

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.condition = Condition()

    def update_frame(self, frame_bytes):
        global _latest_frame
        with output_lock:
            _latest_frame = frame_bytes
        with self.condition:
            self.condition.notify_all()

    def get_frame(self):
        global _latest_frame
        with self.condition:
            self.condition.wait()
        with output_lock:
            return _latest_frame

class StreamingHandler(server.BaseHTTPRequestHandler):
    PAGE = """\
<html>
<head>
  <title>Person Detector Streaming</title>
  <style>
    body { margin: 0; background-color: #000; color: #fff; text-align: center; }
    h1 { margin: 20px; }
    img { width: 100vw; max-height: calc(100vh - 80px); object-fit: contain; }
  </style>
</head>
<body>
  <h1>Person Detector Streaming</h1>
  <img src="stream.mjpg" alt="Live stream"/>
</body>
</html>
"""
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = self.PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    frame = self.server.output.get_frame()
                    if frame is not None:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, output):
        self.output = output
        super().__init__(server_address, RequestHandlerClass)
