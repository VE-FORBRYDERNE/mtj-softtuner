from . import core

import jax
from typing import Callable, TypeVar

T = TypeVar("T", bound=Callable)


def core_dummy_initialized(f: T) -> T:
    def decorated(*a, **k):
        initialized_orig = core.initialized
        core.initialized = True
        try:
            r = f(*a, **k)
        finally:
            core.initialized = initialized_orig
        return r

    return decorated


def core_partly_initialized(f: T) -> T:
    def decorated(*a, **k):
        initialized_orig = core.thread_resources_initialized
        env_orig = jax.experimental.maps.thread_resources.env
        core.initialize_thread_resources(1, backend="cpu")
        try:
            r = f(*a, **k)
        finally:
            core.thread_resources_initialized = initialized_orig
            jax.experimental.maps.thread_resources.env = env_orig
        return r

    return decorated


def core_fully_initialized(f: T) -> T:
    return core_dummy_initialized(core_partly_initialized(f))
