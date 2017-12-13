from os.path import abspath, join, dirname, pardir

SRC_PATH = join(abspath(dirname(__file__)))
CONFIG_PATH = join(SRC_PATH, pardir, 'GOL_config.ini')
