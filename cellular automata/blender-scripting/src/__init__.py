import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "cellular automata/blender-scripting"
sys.path.append(str(SRC_PATH))

CONFIG_PATH = str(SRC_PATH / 'GOL_config.ini')

sys.path.append(str(UTILS_PATH))
import utils.blender_utils
import importlib
importlib.reload(utils.blender_utils)