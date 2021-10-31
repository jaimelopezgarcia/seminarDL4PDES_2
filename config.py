import os

_root = os.path.dirname(__file__)

_data_dir = os.path.join(_root,"data")
_results_dir = os.path.join(_root,"results")
_logs_dir = os.path.join(_root,"logs")

_dirs = [_data_dir, _results_dir, _logs_dir]

for _dir in _dirs:
    try:
        os.makedirs(_dir)
    except:
        pass

settings = {"dir_data": _data_dir,
           "dir_results": _results_dir,
           "dir_logs": _logs_dir}