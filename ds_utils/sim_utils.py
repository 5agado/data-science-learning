from collections import namedtuple
from itertools import starmap, product


def named_configs(items):
    Config = namedtuple('Config', items.keys())
    return starmap(Config, product(*items.values()))