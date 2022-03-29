import os
import pickle


def save_variable(universe: int, name: str, val) -> None:
    """Save a variable so it can be restored later with `restore_variable`."""
    if universe is None:
        return
    universe = int(universe)
    name = str(name)
    folder = os.path.join(".mtjs" + str(universe))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "wb") as f:
        pickle.dump(val, f)


def restore_variable(universe: int, name: str):
    """Restore a variable saved with `save_variable` if it exists."""
    universe = int(universe)
    name = str(name)
    folder = os.path.join(".mtjs" + str(universe))
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        raise ValueError(path)
    with open(path, "rb") as f:
        return pickle.load(f)
