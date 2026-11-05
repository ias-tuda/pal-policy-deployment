"""The voice task list (config/voice_tasks.yaml): the tasks the voice agent may start."""


class VoiceTask(object):
    __slots__ = ("task", "description", "aliases")

    def __init__(self, task, description, aliases=()):
        self.task = task
        self.description = description
        self.aliases = list(aliases)

    def to_dict(self):
        return {"task": self.task, "description": self.description, "aliases": self.aliases}


class VoiceTasks(object):
    """The voice tasks in file order; find() looks a task name up, ignoring case."""

    def __init__(self, tasks):
        self.tasks = list(tasks)
        self._by_name = {task.task.lower(): task for task in self.tasks}

    def find(self, name):
        if not isinstance(name, str):
            return None
        return self._by_name.get(name.strip().lower())

    def names(self):
        return [task.task for task in self.tasks]


def parse_tasks(data):
    """VoiceTasks from the parsed YAML; raise ValueError on a malformed list."""
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list) or not data["tasks"]:
        raise ValueError("the voice task file needs a non-empty list 'tasks'")
    tasks = []
    seen = set()
    for index, entry in enumerate(data["tasks"]):
        if not isinstance(entry, dict):
            raise ValueError("tasks[{}] is not a mapping".format(index))
        task = entry.get("task")
        description = entry.get("description")
        aliases = entry.get("aliases") or []
        if not isinstance(task, str) or not task.strip():
            raise ValueError("tasks[{}] needs a task name".format(index))
        if not isinstance(description, str) or not description.strip():
            raise ValueError("task {} needs a description".format(task))
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            raise ValueError("the aliases of task {} must be a list of strings".format(task))
        if task.strip().lower() in seen:
            raise ValueError("task {} is listed twice".format(task))
        seen.add(task.strip().lower())
        tasks.append(VoiceTask(task.strip(), description.strip(), [alias.strip() for alias in aliases]))
    return VoiceTasks(tasks)


def load_tasks(path):
    import yaml  # Only here, so that the tests of the other modules run without PyYAML.
    with open(path, encoding="utf-8") as stream:
        return parse_tasks(yaml.safe_load(stream))
