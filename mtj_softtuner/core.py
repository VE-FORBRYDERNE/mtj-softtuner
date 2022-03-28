from . import _patch as _
from . import exceptions

import os
import termcolor
import requests  # Only for connecting to Colab TPU and for nothing else
import progressbar
from tqdm.auto import tqdm
import multiprocessing
import time
import json
import zipfile
import math
import jax
import jax.numpy as jnp
import jax.dlpack
import numpy as np
import haiku as hk
import torch
import packaging.version
from typing import Optional
import mesh_transformer
import mesh_transformer.util
import mesh_transformer.layers
import mesh_transformer.transformer_shard
import transformers


DEMATERIALIZED_LOADING_SUPPORTED = hasattr(
    mesh_transformer.transformer_shard, "compute_placeholder_params"
)

initialized = False


def jax_from_dlpack(capsule) -> jnp.array:
    return jax.dlpack.from_dlpack(
        capsule,
        **(
            {}
            if packaging.version.parse(jax.__version__)
            >= packaging.version.parse("0.2.18")
            else {"backend": jax.lib.xla_bridge.get_backend("cpu")}
        ),
    ).copy()


def shatter(in_axes: str, out_axes: str):
    """Helper function for setting up JAX xmaps.

    This is a decorator that creates an xmapped version of a function.
    Your function's arguments should be NumPy arrays or JAX pytrees with
    NumPy arrays at the leaves.  Your actual function will be run 8 times in
    parallel (once for each TPU core).  You can specify for some of your
    function's arguments to be sharded (using a different value for that
    argument on each TPU).  Sharded arguments will be split along the leading
    dimension into 8 subarrays, for example if your sharded argument is an
    array with shape (8, 10, 2), each of the 8 versions of your function will
    receive one subarray with shape (10, 2).  If the leading dimension of your
    sharded arguments aren't equal to 8, you will receive an error.
    The return value(s) of your function can also be sharded, which will
    result in each of the 8 values from your 8 versions of your function to be
    concatenated together along a new leading axis.  Non-sharded arguments
    should have a leading axis of size 1.

    Note: Your function must have at least one sharded argument AND one
    non-sharded argument, otherwise an error will be thrown.  Also your
    function shouldn't have any default arguments, *args or **kwargs.

    Parameters
    ----------
    in_axes : str
        A string with the same length as the number of parameters your function
        has, where each character of the string is 's' or 'b'.  's' means the
        corresponding parameter of your function should be sharded; 'b' means
        the corresponding parameter of your function should not be sharded.
    out_axes : str
        A string with the same length as the number of returns your function
        has, where each character of the string is 's' or 'b'.

    Returns
    -------
    Callable[Callable[..., Any], Callable[..., Any]]
        A function that takes one argument (the function that you want to be
        xmapped) and returns the xmapped version of your function.
    """
    in_axes = tuple(map(lambda c: ["batch" if c == "b" else "shard", ...], in_axes))
    out_axes = tuple(map(lambda c: ["batch" if c == "b" else "shard", ...], out_axes))
    if len(in_axes) == 1:
        in_axes = in_axes[0]
    if len(out_axes) == 1:
        out_axes = out_axes[0]
    return lambda fun: jax.experimental.maps.xmap(
        fun=fun,
        in_axes=in_axes,
        out_axes=out_axes,
        donate_argnums=(0,),
        axis_resources={"shard": "mp", "batch": "dp"},
    )


def shatter(in_axes: str, out_axes: str):
    """Helper function for setting up JAX xmaps.

    This is a decorator that creates an xmapped version of a function.
    Your function's arguments should be NumPy arrays or JAX pytrees with
    NumPy arrays at the leaves.  Your actual function will be run 8 times in
    parallel (once for each TPU core).  You can specify for some of your
    function's arguments to be sharded (using a different value for that
    argument on each TPU).  Sharded arguments will be split along the leading
    dimension into 8 subarrays, for example if your sharded argument is an
    array with shape (8, 10, 2), each of the 8 versions of your function will
    receive one subarray with shape (10, 2).  If the leading dimension of your
    sharded arguments aren't equal to 8, you will receive an error.
    The return value(s) of your function can also be sharded, which will
    result in each of the 8 values from your 8 versions of your function to be
    concatenated together along a new leading axis.  Non-sharded arguments
    should have a leading axis of size 1.

    Note: Your function must have at least one sharded argument AND one
    non-sharded argument, otherwise an error will be thrown.  Also your
    function shouldn't have any default arguments, *args or **kwargs.

    Parameters
    ----------
    in_axes : str
        A string with the same length as the number of parameters your function
        has, where each character of the string is 's' or 'b'.  's' means the
        corresponding parameter of your function should be sharded; 'b' means
        the corresponding parameter of your function should not be sharded.
    out_axes : str
        A string with the same length as the number of returns your function
        has, where each character of the string is 's' or 'b'.

    Returns
    -------
    Callable[Callable[..., Any], Callable[..., Any]]
        A function that takes one argument (the function that you want to be
        xmapped) and returns the xmapped version of your function.
    """
    in_axes = tuple(map(lambda c: ["batch" if c == "b" else "shard", ...], in_axes))
    out_axes = tuple(map(lambda c: ["batch" if c == "b" else "shard", ...], out_axes))
    if len(in_axes) == 1:
        in_axes = in_axes[0]
    if len(out_axes) == 1:
        out_axes = out_axes[0]
    return lambda fun: jax.experimental.maps.xmap(
        fun=fun,
        in_axes=in_axes,
        out_axes=out_axes,
        donate_argnums=(0,),
        axis_resources={"shard": "mp", "batch": "dp"},
    )


class EmbeddingShard(mesh_transformer.transformer_shard.EmbeddingShard):
    """
    A version of Mesh Transformer JAX's EmbeddingShard with a trainable
    soft prompt module
    """

    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        self.softtune_in_dim = EmbeddingShard.soft_in_dim
        self.softtune_in_dim_per_shard = math.ceil(
            self.softtune_in_dim / config["cores_per_replica"]
        )
        self.softtune_proj = hk.Linear(
            self.out_dim,
            w_init=hk.initializers.TruncatedNormal(
                stddev=1 / np.sqrt(self.softtune_in_dim)
            ),
            with_bias=False,
            name="softtune_linear",
        )

    def __call__(self, x: jnp.array, **kwargs) -> jnp.array:
        pe_length = kwargs.get("pe_length", 0)
        pe_length = jnp.int32(pe_length)
        shard_start_index = jax.lax.axis_index("shard") * self.in_dim_per_shard
        proj_out = self.proj(
            jax.nn.one_hot(x - shard_start_index, self.in_dim_per_shard)
        )
        mask = jnp.broadcast_to((x < self.in_dim)[:, jnp.newaxis], proj_out.shape)
        proj_out = jnp.where(mask, proj_out, 0)
        if not kwargs.get("mtj_softtuner_disable_pe", False) and getattr(
            self, "has_sqrt_embed_scale", False
        ):
            proj_out *= jnp.sqrt(self.out_dim).astype(proj_out.dtype)
        if (
            not kwargs.get("mtj_softtuner_disable_pe", False)
            and not getattr(self, "post_embed", False)
            and self.positional_embeddings is not None
        ):
            shard_roll_index = jnp.int32(
                jax.lax.axis_index("shard") * self.out_dim_per_shard
            )
            pos_embed = jnp.pad(
                self.positional_embeddings,
                ((0, 0), (0, self.out_dim - self.out_dim_per_shard)),
            )
            pos_embed = jnp.roll(pos_embed, shard_roll_index, axis=1)
            pos_embed = jnp.roll(
                pos_embed, -pe_length - getattr(self, "pe_shift", 0), axis=0
            )[-proj_out.shape[0] :]
            proj_out += pos_embed
        soft_shard_start_index = (
            jax.lax.axis_index("shard") * self.softtune_in_dim_per_shard
        )
        soft_out = self.softtune_proj(
            jax.nn.one_hot(
                x - soft_shard_start_index - self.in_dim,
                self.softtune_in_dim_per_shard,
            )
        )
        if getattr(self, "has_sqrt_embed_scale", False):
            soft_out *= jnp.sqrt(self.out_dim).astype(proj_out.dtype)
        proj_out += soft_out
        proj_out = mesh_transformer.util.g_psum(proj_out)
        if not kwargs.get("mtj_softtuner_disable_pe", False) and getattr(
            self, "post_embed", False
        ):
            pos_embed = self.positional_embeddings
            pos_embed = jnp.roll(pos_embed, -pe_length, axis=0)[-proj_out.shape[0] :]
            proj_out += pos_embed
        return proj_out


mesh_transformer.transformer_shard.EmbeddingShard = EmbeddingShard


class EmbeddingCausalTransformer(mesh_transformer.transformer_shard.CausalTransformer):
    """
    A version of Mesh Transformer JAX's CausalTransformer with a function for
    embedding a 1D array of token IDs and returning the embedding matrix
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        @shatter("sb", "b")
        def _get_embedding_matrix(params: dict, tokens: jnp.array) -> jnp.array:
            @hk.without_apply_rng
            @hk.transform
            def inner(tokens: jnp.array):
                transformer = mesh_transformer.transformer_shard.CausalTransformerShard(
                    self.config
                )
                return transformer.embed(tokens, mtj_softtuner_disable_pe=True)

            return inner.apply(params, tokens)

        self._get_embedding_matrix = _get_embedding_matrix

    def get_embedding_matrix(self, tokens: np.array) -> jnp.array:
        """Embeds the given array of tokens.

        Parameters
        ----------
        tokens : numpy.array
            A 1-dimensional NumPy/jax.numpy array with dtype `numpy.uint32` or
            `jax.numpy.uint32` containing the IDs of the tokens you want to
            embed.

        Returns
        -------
        jax.numpy.array
            Embedding matrix for your tokens as a 2-dimensional jax.numpy array
            with dtype `jax.numpy.float32` and shape `(len(tokens), d_model)`,
            where `d_model` is the embedding dimension (or "model dimension")
            of your model.
        """
        return self._get_embedding_matrix(
            self.state["params"],
            tokens[np.newaxis, :],
        )[0]


def read_ckpt_custom(
    soft_in_dim: int,
    move_xmap,
    params,
    pytree,
    dir: str,
    shards_in: int,
    load_opt: bool = True,
):
    """Loads the model's state from a checkpoint.

    Parameters
    ----------
    soft_in_dim
        Number of tokens in the soft prompt as a positive integer.
    move_xmap
        network.move_xmap function.
    params
        Network configuration (network.config).
    pytree
        State of the network (network.state).
    dir : str
        Path to the model checkpoint.  Must contain a trailing slash.
    shards_in : int
        Number of shards the model is broken up into.  Should be 8.
    load_opt : bool, default=True
        Whether or not to load the optimizer state from the model checkpoint
        if it has one.

    Returns
    -------
    Any
        Updated version of network.state.
    spmodule : str
        The trainable soft prompt module's module name, so that you can change
        the embedding matrix for the trainable soft prompt (assuming your
        embedding matrix is called `soft_embeddings`) by doing
        `network.state["params"][spmodule]["w"] = soft_embeddings`.
    """
    if not dir.endswith("/"):
        dir += "/"

    pieces = 16

    old_flattened, structure = jax.tree_flatten(pytree)

    soft_embeddings_mask, _ = jax.tree_flatten(
        hk.data_structures.map(
            lambda module_name, name, value: module_name.split("/~/", 3)[-1]
            == "softtune_linear",
            pytree["params"],
        )
    )
    assert sum(soft_embeddings_mask) == 1

    desync_mask, _ = jax.tree_flatten(
        hk.data_structures.map(
            lambda module_name, name, value: module_name.split("/~/", 3)[-1].startswith(
                "replicated_layer_norm"
            ),
            pytree["params"],
        )
    )

    original_opt_state = pytree["opt_state"]

    n_tensors = 0
    for file_index in range(pieces):
        n_tensors += len(np.load(f"{dir}shard_0/{file_index}.npz").keys())

    def _unshard(bar):
        unsharded = []
        tensor_index = progress_index = 0

        for file_index in range(pieces):
            array_keys = [*np.load(f"{dir}shard_0/{file_index}.npz").keys()]
            for array_index in range(len(array_keys)):
                unstacked = []
                for shard_index in range(shards_in):
                    if (
                        tensor_index < len(desync_mask)
                        and desync_mask[tensor_index]
                        and shard_index > 0
                    ):
                        continue
                    if (
                        tensor_index < len(soft_embeddings_mask)
                        and soft_embeddings_mask[tensor_index]
                    ):
                        unsharded.append(
                            jnp.empty(
                                (
                                    shards_in,
                                    math.ceil(soft_in_dim / shards_in),
                                    params["d_model"],
                                ),
                                dtype=jnp.float32,
                            )
                        )
                        tensor_index += 1

                    npz = np.load(f"{dir}shard_{shard_index}/{file_index}.npz")
                    array = npz[array_keys[array_index]]
                    if array.dtype == "V2":
                        array.dtype = jnp.bfloat16
                    unstacked.append(array)

                if tensor_index < len(desync_mask) and desync_mask[tensor_index]:
                    x = move_xmap(
                        jnp.tile(unstacked[0], (shards_in, 1)),
                        np.zeros(params["cores_per_replica"]),
                    )
                else:
                    x = move_xmap(
                        jnp.stack(unstacked),
                        np.zeros(params["cores_per_replica"]),
                    )
                unsharded.append(x)

                bar.update(progress_index)

                assert (
                    x.shape == old_flattened[tensor_index].shape
                ), f"Incompatible checkpoints {x.shape} vs {old_flattened[tensor_index].shape}"
                progress_index += 1
                tensor_index += 1

        return unsharded

    print(
        "\nPlease wait while we load the model's tensors into the TPU memory.",
        flush=True,
    )
    with progressbar.ProgressBar(
        max_value=n_tensors,
        widgets=[
            progressbar.AnimatedMarker(
                "⡀⡁⡂⡃⡄⡅⡆⡇⡈⡉⡊⡋⡌⡍⡎⡏⡐⡑⡒⡓⡔⡕⡖⡗⡘⡙⡚⡛⡜⡝⡞⡟⡠⡡⡢⡣⡤⡥⡦⡧⡨⡩⡪⡫⡬⡭⡮⡯⡰⡱⡲⡳⡴⡵⡶⡷⡸⡹⡺⡻⡼⡽⡾⡿⢀⢁⢂⢃⢄⢅⢆⢇⢈⢉⢊⢋⢌⢍⢎⢏⢐⢑⢒⢓⢔⢕⢖⢗⢘⢙⢚⢛⢜⢝⢞⢟⢠⢡⢢⢣⢤⢥⢦⢧⢨⢩⢪⢫⢬⢭⢮⢯⢰⢱⢲⢳⢴⢵⢶⢷⢸⢹⢺⢻⢼⢽⢾⢿⣀⣁⣂⣃⣄⣅⣆⣇⣈⣉⣊⣋⣌⣍⣎⣏⣐⣑⣒⣓⣔⣕⣖⣗⣘⣙⣚⣛⣜⣝⣞⣟⣠⣡⣢⣣⣤⣥⣦⣧⣨⣩⣪⣫⣬⣭⣮⣯⣰⣱⣲⣳⣴⣵⣶⣷⣸⣹⣺⣻⣼⣽⣾⣿"
            ),
            "  ",
            progressbar.ETA(),
            "   ",
            progressbar.Counter(format=f"%(value){len(str(n_tensors))}d"),
            f"/{n_tensors}  ",
            progressbar.Percentage(),
            "  ",
            progressbar.Bar(left="[", right="]", marker="█"),
        ],
    ) as bar:
        try:
            unsharded = _unshard(bar)
        except AssertionError:
            load_opt = False
            del pytree["opt_state"]
            old_flattened, structure = jax.tree_flatten(pytree)
            unsharded = _unshard(bar)

    loaded_pytree = jax.tree_unflatten(structure, unsharded)

    print("\nFinished loading the model!\n\n\n")

    if not load_opt:
        loaded_pytree["opt_state"] = original_opt_state
    return loaded_pytree, next(
        state
        for state in loaded_pytree["params"]
        if state.split("/~/", 3)[-1] == "softtune_linear"
    )


def reshard_reverse(x, total_shards, old_shape):
    assert len(x.shape) != 1
    if len(x.shape) == 2:
        if old_shape[1] == x.shape[1]:
            out = x[0:1].tile((total_shards, 1))
        else:
            out = x.reshape(old_shape)
    elif len(x.shape) == 3:
        if x.shape[0] * x.shape[2] == old_shape[2]:
            out = x.reshape(old_shape)
        elif x.shape[0] * x.shape[1] == old_shape[1]:
            out = x.reshape((old_shape[1], old_shape[0], old_shape[2])).permute(
                (1, 0, 2)
            )
        else:
            assert False
    else:
        assert False
    return out


def get_hf_conversion_callback(network, model_spec):
    def hf_conversion_callback(model_dict, f, **_):
        with zipfile.ZipFile(f, "r") as z:
            try:
                last_storage_key = None
                f = None
                for key in tqdm(
                    sorted(
                        model_dict.keys(),
                        key=lambda k: (model_dict[k].key, model_dict[k].seek_offset),
                    ),
                    desc="Loading model tensors",
                ):

                    if key not in model_spec:
                        model_dict[key] = torch.empty(
                            model_dict[key].shape,
                            dtype=model_dict[key].dtype,
                            device="meta",
                        )
                        continue

                    storage_key = model_dict[key].key
                    if storage_key != last_storage_key:
                        last_storage_key = storage_key
                        if isinstance(f, zipfile.ZipExtFile):
                            f.close()
                        f = z.open(f"archive/data/{storage_key}")
                    current_offset = f.tell()
                    if current_offset != model_dict[key].seek_offset:
                        f.seek(model_dict[key].seek_offset)
                    spec = model_spec[key]
                    transforms = set(spec.get("transforms", ()))
                    tensor = model_dict[key].materialize(f, map_location="cpu")
                    model_dict[key] = tensor.to("meta")

                    if "divide_by_shards" in transforms:
                        tensor /= network.config["cores_per_replica"]
                    if "vocab_pad" in transforms:
                        tensor = torch.nn.functional.pad(
                            tensor, (0, 0, 0, network.config["n_vocab_padding"])
                        )
                    if "no_transpose" not in transforms and tensor.ndim == 2:
                        tensor = tensor.T
                    tensor.unsqueeze_(0)
                    if tensor.dtype is torch.float16 or tensor.dtype is torch.float32:
                        tensor = tensor.bfloat16()

                    network.state["params"][spec["module"]][
                        spec["param"]
                    ] = network.move_xmap(
                        jax_from_dlpack(
                            torch.utils.dlpack.to_dlpack(
                                reshard_reverse(
                                    tensor,
                                    network.config["cores_per_replica"],
                                    network.state["params"][spec["module"]][
                                        spec["param"]
                                    ].shape,
                                )
                            )
                        ),
                        np.empty(network.config["cores_per_replica"]),
                    )
                for mv in network.state["params"].values():
                    for pk, pv in mv.items():
                        if isinstance(
                            pv, mesh_transformer.transformer_shard.PlaceholderTensor
                        ):
                            mv[pk] = network.move_xmap(
                                jnp.zeros(mv[pk].shape, dtype=jnp.bfloat16),
                                np.empty(network.config["cores_per_replica"]),
                            )
            except Exception as e:
                import traceback

                traceback.print_exc()
                print("ERROR: ", e)
                raise e
            finally:
                if isinstance(f, zipfile.ZipExtFile):
                    f.close()

    return hf_conversion_callback


@shatter("sb", "s")
def _init_opt_state(params, aux: jnp.array):
    return mesh_transformer.util.to_f32(_init_opt_state.optimizer.init(params))


def init_opt_state(params, optimizer):
    """Returns initialized optax state for the given haiku parameters and optax optimizer"""
    _init_opt_state.optimizer = optimizer
    return _init_opt_state(params, np.empty(1))


def show_spinner() -> multiprocessing.Process:
    """
    Shows a bouncing progress bar.  To stop it, save the return value of this
    function as (for example) `spinner`, and then run spinner.terminate().
    """

    def _show_spinner():
        bar = progressbar.ProgressBar(
            max_value=progressbar.UnknownLength,
            widgets=[
                progressbar.Timer(),
                "  ",
                progressbar.BouncingBar(left="[", right="]", marker="█"),
            ],
        )
        i = 0
        while True:
            bar.update(i)
            time.sleep(0.1)
            i += 1

    spinner = multiprocessing.Process(target=_show_spinner, args=())
    spinner.start()
    return spinner


def initialize(quiet=exceptions.default_quiet):
    global initialized
    if initialized:
        return
    print(
        termcolor.colored("\n\nConnecting to your Colab instance's TPU...", "magenta"),
        flush=True,
    )
    spinner = show_spinner()
    colab_tpu_addr = os.environ["COLAB_TPU_ADDR"].split(":")[0]
    requests.post(
        f"http://{colab_tpu_addr}:8475/requestversion/tpu_driver0.1_dev20210607"
    )
    jax.config.FLAGS.jax_xla_backend = "tpu_driver"
    jax.config.FLAGS.jax_backend_target = "grpc://" + os.environ["COLAB_TPU_ADDR"]
    spinner.terminate()
    print(flush=True)
    if jax.device_count() < 8:
        raise exceptions.ConfigurationError(
            "We couldn't detect your Colab instance's TPU.\nTry restarting the runtime (Runtime > Restart Runtime) and trying again.",
            code=2,
            quiet=quiet,
        )
    initialized = True


def initialize_thread_resources(shards: int):
    mesh_shape = (1, shards)
    devices = np.array(jax.devices()[:shards]).reshape(mesh_shape)
    thread_resources_env = jax.experimental.maps.ResourceEnv(
        jax.experimental.maps.Mesh(devices, ("dp", "mp")),
        *(
            ((),)
            if packaging.version.parse(jax.__version__)
            >= packaging.version.parse("0.2.15")
            else ()
        ),
    )
    jax.experimental.maps.thread_resources.env = thread_resources_env
    return thread_resources_env


def get_tokenizer(params: dict, tokenizer_id: Optional[str] = None):
    return transformers.GPT2TokenizerFast.from_pretrained(
        params.get("tokenizer_id", "gpt2")
    )


def get_hf_checkpoint_metadata(ckpt_path: str):
    ckpt_path = ckpt_path.rstrip("/")
    model_config = transformers.AutoConfig.from_pretrained(ckpt_path)

    spec_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "kobold",
        "maps",
        model_config.model_type + ".json",
    )
    if not os.path.isfile(spec_path):
        raise NotImplementedError(
            f"Unsupported model type {repr(model_config.model_type)}"
        )
    with open(spec_path) as f:
        lazy_load_spec = json.load(f)
    params = model_config.to_diff_dict()

    if "mtj_compat" in lazy_load_spec:
        params["compat"] = lazy_load_spec["mtj_compat"]
    if "mtj_pe" in lazy_load_spec:
        params["pe"] = lazy_load_spec["mtj_pe"]
    for k, v in lazy_load_spec.get("mtj_config_map", {}).items():
        if type(v) is not list:
            params[k] = params[v]
            continue
        for i in range(len(v)):
            if i == len(v) - 1:
                params[k] = v[i]
            elif v[i] in params:
                params[k] = params[v[i]]
                break

    model_spec = {}
    for key, spec in lazy_load_spec.get("static_weights", {}).items():
        if spec.get("mtj") is not None:
            model_spec[key] = spec["mtj"].copy()
            model_spec[key]["module"] = (
                "causal_transformer_shard/~/" + model_spec[key]["module"]
            )
    for _key, spec in lazy_load_spec.get("layer_weights", {}).items():
        for layer in range(params["layers"]):
            if spec.get("mtj") is not None:
                key = _key.format(layer=layer)
                model_spec[key] = spec["mtj"].copy()
                model_spec[key]["module"] = "causal_transformer_shard/~/" + model_spec[
                    key
                ]["module"].format(layer=layer)

    params["n_vocab"] = params["vocab_size"]

    if "activation_function" in params:
        params["activation"] = params["activation_function"]

    params["norm"] = "layernorm"

    for c in (8, 6, 4, 2, 1):
        if 0 == params["n_heads"] % c == params["d_model"] % c:
            params["cores_per_replica"] = c
            break

    params["n_vocab_padding"] = -(params["n_vocab"] % -params["cores_per_replica"])
    params["tokenizer_id"] = ckpt_path
    tokenizer = get_tokenizer(params)
    newlinemode = params.get(
        "newlinemode", "s" if model_config.model_type == "xglm" else "n"
    )
    params["max_batch_size"] = 450 if params["d_model"] > 4096 else 2048
    params["eos_token"] = (
        [50259, 50259] if model_config.model_type == "xglm" else [50256]
    )
    params["seq"] = 2048
    return lazy_load_spec, model_spec, params, tokenizer, newlinemode
