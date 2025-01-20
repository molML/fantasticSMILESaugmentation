import json
import pathlib

def get_package_path():
    return str(pathlib.Path(__file__).parent.parent.resolve())


package_path = get_package_path()

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
