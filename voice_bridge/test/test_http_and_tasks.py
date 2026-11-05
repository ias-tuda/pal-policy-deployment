"""The HTTP API and the voice task file."""
import json
import os
import threading
import unittest
import urllib.error
import urllib.request

from helpers import Rig

from voice_bridge.http_api import make_server
from voice_bridge.tasks import load_tasks, parse_tasks

TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "voice_tasks.yaml")

try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


class HttpApi(unittest.TestCase):

    def setUp(self):
        self.rig = Rig()
        self.server = make_server(self.rig.bridge, "127.0.0.1", 0, {"/sim": lambda body: {"echo": body}})
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:{}".format(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.rig.close()

    def request(self, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request = urllib.request.Request(self.url + path, data=data)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode())

    def test_get_status_and_tasks(self):
        code, status = self.request("/status")
        self.assertEqual((code, status["task"]), (200, "grasp-cube"))
        code, tasks = self.request("/tasks")
        self.assertEqual((code, len(tasks)), (200, 7))

    def test_start_task(self):
        code, result = self.request("/start_task", {"task": "stack-cube"})
        self.assertEqual(code, 200)
        self.assertEqual((result["ok"], result["code"], result["task"]), (True, "started", "stack-cube"))

    def test_a_refusal_is_a_normal_answer(self):
        code, result = self.request("/start_task", {"task": "make-coffee"})
        self.assertEqual((code, result["ok"], result["code"]), (200, False, "unknown_task"))

    def test_malformed_requests(self):
        self.assertEqual(self.request("/start_task", raw=b"{not json")[0], 400)
        self.assertEqual(self.request("/start_task", {"name": "stack-cube"})[0], 400)
        self.assertEqual(self.request("/start_task", raw=b"")[0], 400)
        self.assertEqual(self.request("/nothing")[0], 404)
        self.assertEqual(self.request("/nothing", {"task": "x"})[0], 404)

    def test_extra_post_path(self):
        self.assertEqual(self.request("/sim", {"a": 1}), (200, {"echo": {"a": 1}}))

    def test_refuses_web_pages(self):
        for data in (None, json.dumps({"task": "stack-cube"}).encode()):
            request = urllib.request.Request(self.url + ("/status" if data is None else "/start_task"), data=data,
                                             headers={"Origin": "http://example.org"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=10)
            self.assertEqual(caught.exception.code, 403)
        self.assertEqual(self.rig.state.task_name(), "grasp-cube")


class TaskFile(unittest.TestCase):

    def test_parse(self):
        tasks = parse_tasks({"tasks": [{"task": "grasp-cube", "description": "Würfel greifen",
                                        "aliases": ["Würfel hoch"]}]})
        self.assertEqual(tasks.find("GRASP-CUBE").aliases, ["Würfel hoch"])
        self.assertIsNone(tasks.find("stack-cube"))
        self.assertIsNone(tasks.find(None))

    def test_malformed(self):
        for data in (None, {}, {"tasks": []}, {"tasks": ["grasp-cube"]},
                     {"tasks": [{"task": "grasp-cube"}]},
                     {"tasks": [{"task": "a", "description": "d", "aliases": "b"}]},
                     {"tasks": [{"task": "a", "description": "d"}, {"task": "A", "description": "e"}]}):
            with self.assertRaises(ValueError):
                parse_tasks(data)

    @unittest.skipUnless(HAVE_YAML, "needs PyYAML")
    def test_the_shipped_file(self):
        tasks = load_tasks(TASKS_FILE)
        self.assertEqual(tasks.names(), ["grasp-cube", "stack-cube", "putintobox-s102-wx-a-h22-s1",
                                         "putballintobox-s15-pb-dr-s1", "mug-ball", "mug-turn", "mug-coaster"])


if __name__ == "__main__":
    unittest.main()
