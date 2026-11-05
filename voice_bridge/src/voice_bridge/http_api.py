"""The HTTP/JSON API for the voice agent; it listens on localhost unless told otherwise.

  GET  /status      state, task, readiness, start pose, objects, busy, last result
  GET  /tasks       [{"task", "description", "aliases", "available"}], the voice tasks
  POST /start_task  body {"task": "<name>"}; answers when the policy runs or the command is
                    refused: {"ok", "code", "message", "task"}

A refused command is a normal answer (HTTP 200, "ok": false); 4xx is a malformed request.
/start_task can take minutes (Settings.total()): give the client a read timeout above it.
A request with an Origin header (a web page in a browser on this PC) is refused with 403.
extra_post adds POST paths: path -> function(body dict) -> JSON data (the simulation's /sim).
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

MAX_BODY = 4096


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """http.server has its own only from Python 3.7; the dev container may have 3.6."""
    daemon_threads = True


def make_server(bridge, host, port, extra_post=None):
    extra_post = extra_post or {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Origin"):
                self._send(403, {"error": "no requests from web pages"})
            elif self.path == "/status":
                self._send(200, bridge.status())
            elif self.path == "/tasks":
                self._send(200, bridge.tasks())
            else:
                self._send(404, {"error": "unknown path {}".format(self.path)})

        def do_POST(self):
            if self.headers.get("Origin"):
                self._send(403, {"error": "no requests from web pages"})
                return
            if self.path != "/start_task" and self.path not in extra_post:
                self._send(404, {"error": "unknown path {}".format(self.path)})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY:
                self._send(400, {"error": "send a JSON body of at most {} bytes".format(MAX_BODY)})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                self._send(400, {"error": "the body is not JSON"})
                return
            if self.path in extra_post:
                if not isinstance(body, dict):
                    self._send(400, {"error": "send a JSON object"})
                    return
                self._send(200, extra_post[self.path](body))
                return
            if not isinstance(body, dict) or not isinstance(body.get("task"), str):
                self._send(400, {"error": 'send {"task": "<name>"}'})
                return
            self._send(200, bridge.start_task(body["task"]).to_dict())

        def _send(self, code, data):
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The client gave up; the command has run to its end anyway.

        def log_message(self, *args):
            pass  # The outcomes go to the supervisor log.

    return ThreadingHTTPServer((host, port), Handler)
