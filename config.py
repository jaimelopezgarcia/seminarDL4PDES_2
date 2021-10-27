import os

_root = os.path.dirname(__file__)

_data_dir = os.path.join(_root,"data")

try:
    os.makedirs(_data_dir)
except:
    pass

settings = {"dir_data": _data_dir}