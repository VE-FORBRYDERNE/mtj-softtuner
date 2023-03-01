from . import core
from .kobold import utils

import os
import logging
import functools
import jax
import packaging.version
import haiku as hk
import transformers
import mesh_transformer
import mesh_transformer.util
import mesh_transformer.layers
import mesh_transformer.transformer_shard
from typing import Any, Callable, Dict, Tuple, TypeVar


__F = TypeVar("__F", bound=Callable)

JAX13 = packaging.version.parse(jax.__version__) >= packaging.version.parse("0.2.13")
patched = False


old_getnorm = mesh_transformer.layers.getnorm


def getnorm(norm_type: str):
    if norm_type == "layernorm":
        return hk.LayerNorm(-1, True, True, name="replicated_layer_norm")
    elif norm_type == "layernorm-nobias":
        return hk.LayerNorm(-1, True, False, name="replicated_layer_norm")
    else:
        return old_getnorm(norm_type)


mesh_transformer.layers.getnorm = getnorm


def patch(f: __F) -> __F:
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        global patched

        if patched:
            return f(*args, **kwargs)

        old_haiku_flatmapping = os.environ.get("HAIKU_FLATMAPPING", "")
        old_jax_tree_map = jax.tree_map

        # Required for certain optax optimizers to work properly with haiku modules
        # as per https://github.com/deepmind/dm-haiku/issues/191
        os.environ["HAIKU_FLATMAPPING"] = "0"

        # In JAX 0.2.13, jax.tree_multimap was renamed to jax.tree_map and
        # jax.tree_multimap became an alias of this new jax.tree_map,
        # and optax depends on this change, so we're shimming jax.tree_map calls to
        # go to the current jax.tree_multimap (which is the same as the JAX 0.2.13
        # versions of both jax.tree_map and jax.tree_multimap) for compatibility
        # with optax.
        jax.tree_map = jax.tree_multimap

        # Colab doesn't seem to like ReplicatedLayerNorm, so we're just going to use
        # haiku's standard LayerNorm modules, which we can do because we aren't going
        # to train any layernorm parameters.
        old_getnorm = mesh_transformer.layers.getnorm

        # If using JAX 0.2.13 or later, disable warnings about these two JAX functions
        # having been renamed
        if JAX13:
            old_jax_host_count = jax.host_count
            old_jax_host_id = jax.host_id
            jax.host_count = jax.process_count
            jax.host_id = jax.process_index

        mesh_transformer.layers.getnorm = getnorm

        # Allows mtj-softtuner to use aria2 to download models

        old_vars = utils.koboldai_vars

        class Vars:
            aria2_port = 6799
            revision = None

        utils.koboldai_vars = Vars

        logger = logging.getLogger("urllib3")
        old_level = logger.level
        logger.setLevel(logging.ERROR)

        old_from_pretrained = transformers.PreTrainedModel.from_pretrained
        _old_from_pretrained = old_from_pretrained.__func__

        @classmethod
        def new_from_pretrained(
            cls, pretrained_model_name_or_path, *model_args, **kwargs
        ):
            utils.num_shards = None
            utils.current_shard = 0
            utils.from_pretrained_model_name = pretrained_model_name_or_path
            utils.from_pretrained_index_filename = None
            utils.from_pretrained_kwargs = kwargs
            utils.bar = None
            if not core.no_aria2:
                utils.aria2_hook(pretrained_model_name_or_path, **kwargs)
            return _old_from_pretrained(
                cls, pretrained_model_name_or_path, *model_args, **kwargs
            )

        transformers.PreTrainedModel.from_pretrained = new_from_pretrained
        if hasattr(transformers.modeling_utils, "get_checkpoint_shard_files"):
            old_get_checkpoint_shard_files = (
                transformers.modeling_utils.get_checkpoint_shard_files
            )

            def new_get_checkpoint_shard_files(
                pretrained_model_name_or_path, index_filename, *args, **kwargs
            ):
                utils.num_shards = utils.get_num_shards(index_filename)
                utils.from_pretrained_index_filename = index_filename
                return old_get_checkpoint_shard_files(
                    pretrained_model_name_or_path, index_filename, *args, **kwargs
                )

            transformers.modeling_utils.get_checkpoint_shard_files = (
                new_get_checkpoint_shard_files
            )

        patched = True

        try:
            r = f(*args, **kwargs)
        finally:
            os.environ["HAIKU_FLATMAPPING"] = old_haiku_flatmapping
            jax.tree_map = old_jax_tree_map
            mesh_transformer.layers.getnorm = old_getnorm
            if JAX13:
                jax.host_count = old_jax_host_count
                jax.host_id = old_jax_host_id
            utils.koboldai_vars = old_vars
            logger.setLevel(old_level)
            transformers.PreTrainedModel.from_pretrained = old_from_pretrained
            if hasattr(transformers.modeling_utils, "get_checkpoint_shard_files"):
                transformers.modeling_utils.get_checkpoint_shard_files = (
                    old_get_checkpoint_shard_files
                )
            patched = False
        return r

    return decorated


class PatchMeta(type):
    def __new__(
        cls,
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        *args,
        **kwargs
    ):
        for attr in namespace:
            value = namespace[attr]
            if callable(value):
                namespace[attr] = patch(value)
        return type.__new__(cls, name, bases, namespace, *args, **kwargs)
