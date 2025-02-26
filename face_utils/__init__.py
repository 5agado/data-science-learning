import os
from os.path import join, dirname
import logging
CONFIG_PATH = join(dirname(__file__), 'configs', 'face_detection.yaml')

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)

# Console handler
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s - %(message)s", "%H:%M:%S")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)