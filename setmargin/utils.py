import pickle
import numpy as np
import logging
from textwrap import dedent


def load(path, **kwargs):
    with open(path, "rb") as fp:
        return pickle.load(fp, **kwargs)


def dump(path, what, **kwargs):
    with open(path, "wb") as fp:
        pickle.dump(what, fp, **kwargs)


def arr2str(a):
    return np.array2string(a, max_line_width=np.inf, separator=',',
        precision=None, suppress_small=None).replace('\n', '')


def dict2str(d):
    return sorted(d.items())


class LazyMessage(object):
    def __init__(self, fmt, kwargs):
        self.fmt = fmt
        self.kwargs = kwargs

    def arg_eval(self, arg):
        if isinstance(arg, np.ndarray):
            return arr2str(arg)
        elif isinstance(arg, dict):
            return dict2str(arg)
        return arg

    def __str__(self):
        _kwargs = {key: self.arg_eval(arg) for key, arg in self.kwargs.items()}
        return self.fmt.format(**_kwargs)


def subdict(d, keys=None, nokeys=None):
    keys = set(keys if keys else d.keys())
    nokeys = set(nokeys if nokeys else [])
    return {k: v for k, v in d.items() if k in (keys - nokeys)}


class MultilogAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msgargs = subdict(kwargs, nokeys={'extra', 'exc_info', 'stack_info'})
            kwargs = subdict(kwargs, keys={'extra', 'exc_info', 'stack_info'})
            msg = dedent(msg).strip()
            for msg in msg.splitlines():
                msg, kwargs = self.process(msg, kwargs)
                self.logger._log(level, LazyMessage(msg, msgargs), (), **kwargs)


def get_logger(name):
    return MultilogAdapter(logging.getLogger(name))


def freeze(x):
    """Freezes a dictionary, i.e. makes it immutable and thus hashable."""
    frozen = {}
    for k, v in x.items():
        if isinstance(v, list):
            frozen[k] = tuple(v)
        else:
            frozen[k] = v
    return frozenset(frozen.items())

