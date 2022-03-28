import os
import jax
import packaging.version
import haiku as hk
import mesh_transformer
import mesh_transformer.util
import mesh_transformer.layers
import mesh_transformer.transformer_shard


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
if packaging.version.parse(jax.__version__) >= packaging.version.parse("0.2.13"):
    jax.host_count = jax.process_count
    jax.host_id = jax.process_index


def getnorm(type: str):
    if type == "layernorm":
        return hk.LayerNorm(-1, True, True, name="replicated_layer_norm")
    elif type == "layernorm-nobias":
        return hk.LayerNorm(-1, True, False, name="replicated_layer_norm")
    else:
        return old_getnorm(type)


mesh_transformer.layers.getnorm = getnorm
