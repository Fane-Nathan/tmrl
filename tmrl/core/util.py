# standard library imports
import functools
from importlib import import_module
import inspect
import json
import logging
import operator
import os
from pathlib import Path
import pickle
import signal
import subprocess
import tempfile
import threading
from typing import TYPE_CHECKING, Callable, Dict, Type, TypeVar, Union
import weakref

if TYPE_CHECKING:
    import pandas as pd


__docformat__ = "google"

T = TypeVar('T')  # helps with type inference in some editors


def pandas_dict(*args, **kwargs) -> "pd.Series":
    """Constructs an object-dtype ``pandas.Series`` from dict arguments."""
    import pandas as pd

    return pd.Series(dict(*args, **kwargs), dtype=object)


def shallow_copy(obj: T) -> T:
    """Creates a shallow copy of ``obj`` without invoking ``__init__``."""
    x = type(obj).__new__(type(obj))
    vars(x).update(vars(obj))
    return x


# === cached property =================================================================================================


# noinspection PyPep8Naming
class cached_property:
    """Similar to ``property`` but caches the result keyed by instance identity.

    Cached values are kept outside the instance's ``__dict__``, which prevents
    transient attributes from being pickled.
    """

    def __init__(self, init=None):
        self.cache = {}
        self.init = init

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if id(instance) not in self.cache:
            if self.init is None:
                raise AttributeError()
            self.__set__(instance, self.init(instance))
        return self.cache[id(instance)][0]

    def __set__(self, instance, value):
        # Cache the attribute value based on instance id; clean up automatically on GC.
        self.cache[id(instance)] = (value, weakref.ref(instance, functools.partial(self.cache.pop, id(instance))))


# === partial ==========================================================================================================


def default():
    raise ValueError("This is a dummy function and not meant to be called.")


def partial(func: Type[T] = default, *args, **kwargs) -> Union[T, Type[T]]:
    """Like ``functools.partial``, except nested dummy ``partial`` kwargs inherit parameter defaults."""
    if func is not default:
        for k, v in kwargs.items():
            if isinstance(v, functools.partial) and v.func is default:
                kwargs[k] = partial(inspect.signature(func).parameters[k].default, *v.args, **v.keywords)
    return functools.partial(func, *args, **kwargs)


FKEY = '+'


def partial_to_dict(p: functools.partial, version="3"):
    """Serializes a ``functools.partial`` (used for wandb config logging)."""
    assert not p.args, "So far only keyword arguments are supported, here"
    fields = {k: v.default for k, v in inspect.signature(p.func).parameters.items()}
    fields = {k: v for k, v in fields.items() if v is not inspect.Parameter.empty}
    fields.update(p.keywords)
    nested = {k: partial_to_dict(partial(v), version="") for k, v in fields.items() if callable(v)}
    simple = {k: v for k, v in fields.items() if k not in nested}
    output = {FKEY: p.func.__module__ + ":" + p.func.__qualname__, **simple, **nested}
    return dict(output, __format_version__=version) if version else output


def get_class_or_function(func):
    """Imports and returns the object specified as ``'module.path:attr_name'``."""
    module, name = func.split(":")
    return getattr(import_module(module), name)


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def partial_from_args(func: Union[str, Callable], kwargs: Dict[str, str]):
    """Builds a ``functools.partial`` by parsing string arguments against ``func``'s signature."""
    func = get_class_or_function(func) if isinstance(func, str) else func
    keys = {k.split('.')[0] for k in kwargs}
    keywords = {}
    for key in keys:
        params = inspect.signature(func).parameters
        assert key in params, f"'{key}' is not a valid parameter of {func}. Valid parameters are {tuple(params.keys())}."
        param = params[key]
        value = kwargs.get(key, param.default)
        if param.annotation is type:
            sub_keywords = {k.split('.', 1)[1]: v for k, v in kwargs.items() if k.startswith(key + '.')}
            keywords[key] = partial_from_args(value, sub_keywords)
        elif param.annotation is bool:
            keywords[key] = _parse_bool(value)
        else:
            keywords[key] = param.annotation(value)
    return partial(func, **keywords)


# === git ==============================================================================================================


def get_output(*args, default='', **kwargs):
    """Runs a subprocess command and returns its stripped stdout, or ``default`` on error."""
    try:
        output = subprocess.check_output(*args, universal_newlines=True, **kwargs)
        return output.rstrip("\n")  # skip trailing newlines as done in bash
    except (subprocess.CalledProcessError, OSError):
        return default


def git_info(path=None):
    """Returns a dict with metadata about the git repository at ``path``."""
    if path is None:
        import __main__
        main_file = getattr(__main__, "__file__", None)
        if main_file and main_file != "<stdin>":
            path = str(Path(main_file).resolve().parent)
        else:
            path = os.getcwd()

    rev = get_output('git rev-parse HEAD'.split(), cwd=path)
    count = int(get_output('git rev-list HEAD --count'.split(), default='-1', cwd=path))
    status = get_output('git status --short'.split(), cwd=path)
    commit_date = get_output(
        "git show --quiet --date=format-local:%Y-%m-%dT%H:%M:%SZ --format=%cd".split(),
        cwd=path,
        env={**os.environ, "TZ": "UTC"},
    )
    desc = get_output(['git', 'describe', '--long', '--tags', '--dirty', '--always', '--match', r'v[0-9]*\.[0-9]*'], cwd=path)
    message = desc + " " + ' '.join(get_output(['git', 'log', '--oneline', '--format=%B', '-n', '1', "HEAD"], cwd=path).splitlines())

    url = get_output('git config --get remote.origin.url'.split(), cwd=path).strip()
    if url.startswith('git@github.com:'):
        repo_path = url.removeprefix('git@github.com:').removesuffix('.git')
        url = f'https://github.com/{repo_path}/commit/{rev}'
    elif url.startswith('https://github.com'):
        url = f"{url.removesuffix('.git')}/commit/{rev}"

    return dict(url=url, rev=rev, count=count, status=status, desc=desc, date=commit_date, message=message)


# === serialization ====================================================================================================


def dump(obj, path):
    """Atomically pickles ``obj`` to ``path`` while delaying termination signals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with DelayInterrupt():
        tmp_file = tempfile.NamedTemporaryFile(
            mode='wb',
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix='.tmp',
            delete=False,
        )
        tmp_path = Path(tmp_file.name)
        try:
            with tmp_file as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise


def load(path):
    """Loads a pickled object from ``path``."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def cloudpickle_dump(obj, path):
    """Atomically serializes ``obj`` with ``cloudpickle`` to ``path`` while delaying signals."""
    import cloudpickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with DelayInterrupt():
        tmp_file = tempfile.NamedTemporaryFile(
            mode='wb',
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix='.tmp',
            delete=False,
        )
        tmp_path = Path(tmp_file.name)
        try:
            with tmp_file as f:
                cloudpickle.dump(obj, f)
            os.replace(tmp_path, path)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise


def cloudpickle_load(path):
    """Loads a ``cloudpickle``-serialized object from ``path``."""
    import cloudpickle

    with open(path, 'rb') as f:
        return cloudpickle.load(f)


def save_json(d, path):
    """Writes ``d`` as formatted UTF-8 JSON to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def load_json(path):
    """Reads and parses a UTF-8 JSON file from ``path``."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# === signal handling ==================================================================================================


class DelayInterrupt:
    """Catches SIGINT and SIGTERM and re-raises KeyboardInterrupt after the context exits.

    Can be used in a context, e.g., ``with DelayInterrupt():``.
    Signal handling is only installed when entered from the main thread.
    """

    signals = (signal.SIGINT, signal.SIGTERM)

    def __init__(self):
        self.signal_received = False
        self.default_handlers = []
        self._enabled = False

    def __enter__(self):
        self.signal_received = False
        self.default_handlers = []
        self._enabled = threading.current_thread() is threading.main_thread()
        if self._enabled:
            self.default_handlers = [signal.getsignal(s) for s in self.signals]
            for s in self.signals:
                signal.signal(s, self.on_signal)
        return self

    def on_signal(self, *args):
        sig = args[0] if args else "unknown"
        logging.info("tmrl.core.util.DelayInterrupt -- Signal received (%s)", sig)
        self.signal_received = True

    def __exit__(self, *args):
        if self._enabled:
            for s, d in zip(self.signals, self.default_handlers):
                if d is not None:
                    signal.signal(s, d)
        if self.signal_received:
            raise KeyboardInterrupt()


# === operations =======================================================================================================


def prod(iterable):
    """Returns the product of elements in ``iterable`` (``1`` for an empty iterable)."""
    return functools.reduce(operator.mul, iterable, 1)
