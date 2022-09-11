from . import patch  # pylint: disable=unused-import
from . import exceptions

import os
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), "kobold"))

from .kobold import utils

import termcolor
import requests  # Only for connecting to Colab TPU and for nothing else
import progressbar
from tqdm.auto import tqdm
import multiprocessing
import contextlib
import functools
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
import torch.utils.dlpack
import packaging.version
from typing import Callable, Optional, TypeVar
import mesh_transformer
import mesh_transformer.util
import mesh_transformer.layers
import mesh_transformer.transformer_shard
import transformers


BACKEND = "kaggle"

DEMATERIALIZED_LOADING_SUPPORTED = hasattr(
    mesh_transformer.transformer_shard, "compute_placeholder_params"
)

test_mode = False
no_aria2 = False
initialized = False
thread_resources_initialized = False


def jax_from_dlpack(capsule) -> jnp.DeviceArray:
    return jax.dlpack.from_dlpack(
        capsule,
        **(
            {}
            if packaging.version.parse(jax.__version__)
            >= packaging.version.parse("0.2.18")
            else {"backend": jax.lib.xla_bridge.get_backend("cpu")}
        ),
    ).copy()


class CoreNotInitializedError(Exception):
    pass


class _ShatterFunction:
    def __init__(self, fun: Callable, in_axes, out_axes):
        self.__sf_fun = fun
        self.__sf_in_axes = in_axes
        self.__sf_out_axes = out_axes
        self.__sf_mapped: Optional[Callable] = None

    def __call__(self, *args, **kwargs):
        if not thread_resources_initialized:
            raise CoreNotInitializedError(
                "Called a @shatter function before `initialize_thread_resources()` was called"
            )
        if self.__sf_mapped is None:
            self.__sf_mapped = jax.experimental.maps.xmap(
                fun=self.__sf_fun,
                in_axes=self.__sf_in_axes,
                out_axes=self.__sf_out_axes,
                donate_argnums=(0,),
                axis_resources={"shard": "mp", "batch": "dp"},
            )
        return self.__sf_mapped(*args, **kwargs)


__F = TypeVar("__F", bound=Callable)


def shatter(in_axes: str, out_axes: str):
    """
    Helper function for setting up JAX xmaps.

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

    def decorator(fun: __F) -> __F:
        return functools.wraps(fun)(_ShatterFunction(fun, in_axes, out_axes))

    return decorator


class EmbeddingShard(mesh_transformer.transformer_shard.EmbeddingShard):

    """
    A version of Mesh Transformer JAX's EmbeddingShard with a trainable
    soft prompt module
    """

    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        self.softtune_in_dim = config["soft_in_dim"]
        self.softtune_in_dim_per_shard = math.ceil(
            self.softtune_in_dim / config["cores_per_replica"]
        )
        self.softtune_proj = hk.Linear(
            getattr(self, "d_embed", self.out_dim),
            w_init=hk.initializers.TruncatedNormal(
                stddev=1 / np.sqrt(self.softtune_in_dim)
            ),
            with_bias=False,
            name="softtune_linear",
        )

    def __call__(self, x: jnp.DeviceArray, **kwargs) -> jnp.DeviceArray:
        pe_length = kwargs.get("pe_length", 0)
        pe_length = jnp.int32(pe_length)
        shard_start_index = jax.lax.axis_index("shard") * self.in_dim_per_shard
        proj_out = self.proj(
            jax.nn.one_hot(x - shard_start_index, self.in_dim_per_shard)
        )
        mask = jnp.broadcast_to((x < self.in_dim)[:, jnp.newaxis], proj_out.shape)
        proj_out = jnp.where(mask, proj_out, 0)
        soft_shard_start_index = (
            jax.lax.axis_index("shard") * self.softtune_in_dim_per_shard
        )
        soft_out = self.softtune_proj(
            jax.nn.one_hot(
                x - soft_shard_start_index - self.in_dim,
                self.softtune_in_dim_per_shard,
            )
        )
        proj_out += soft_out
        if not kwargs.get("mtj_softtuner_disable_pe", False) and getattr(
            self, "has_sqrt_embed_scale", False
        ):
            proj_out *= jnp.sqrt(self.out_dim).astype(proj_out.dtype)
        if (
            not kwargs.get("mtj_softtuner_disable_pe", False)
            and getattr(self, "project_in") is not None
        ):
            proj_out @= self.project_in
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
        proj_out = mesh_transformer.util.g_psum(proj_out)
        if not kwargs.get("mtj_softtuner_disable_pe", False) and getattr(
            self, "post_embed", False
        ):
            pos_embed = self.positional_embeddings
            pos_embed = jnp.roll(pos_embed, -pe_length, axis=0)[-proj_out.shape[0] :]
            proj_out += pos_embed
        if not kwargs.get("mtj_softtuner_disable_pe", False) and hasattr(self, "norm"):
            proj_out = mesh_transformer.util.f_psum(proj_out)
            proj_out = self.norm(proj_out)
        return proj_out


mesh_transformer.transformer_shard.EmbeddingShard = EmbeddingShard


class EmbeddingCausalTransformer(
    mesh_transformer.transformer_shard.CausalTransformer,
    metaclass=patch.PatchMeta,
):
    """
    A version of Mesh Transformer JAX's CausalTransformer with a function for
    embedding a 1D array of token IDs and returning the embedding matrix
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        @shatter("sb", "b")
        def _get_embedding_matrix(
            params: dict, tokens: jnp.DeviceArray
        ) -> jnp.DeviceArray:
            @hk.without_apply_rng
            @hk.transform
            def inner(tokens: jnp.DeviceArray):
                transformer = mesh_transformer.transformer_shard.CausalTransformerShard(
                    self.config
                )
                return transformer.embed(tokens, mtj_softtuner_disable_pe=True)

            return inner.apply(params, tokens)

        self._get_embedding_matrix = _get_embedding_matrix

    def get_embedding_matrix(self, tokens: np.ndarray) -> jnp.DeviceArray:
        """
        Embeds the given array of tokens.

        Parameters
        ----------
        tokens : numpy.ndarray
            A 1-dimensional NumPy/jax.numpy array with dtype `numpy.uint32` or
            `jax.numpy.uint32` containing the IDs of the tokens you want to
            embed.

        Returns
        -------
        jax.numpy.DeviceArray
            Embedding matrix for your tokens as a 2-dimensional jax.numpy array
            with dtype `jax.numpy.float32` and shape `(len(tokens), d_embed)`,
            where `d_embed` is the embedding dimension of your model.
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
    ckpt_dir: str,
    shards_in: int,
    load_opt: bool = True,
):
    """
    Loads the model's state from a checkpoint.

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
    ckpt_dir : str
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
    if not ckpt_dir.endswith("/"):
        ckpt_dir += "/"

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
        n_tensors += len(np.load(f"{ckpt_dir}shard_0/{file_index}.npz").keys())

    def _unshard(bar):
        unsharded = []
        tensor_index = progress_index = 0

        for file_index in range(pieces):
            array_keys = [*np.load(f"{ckpt_dir}shard_0/{file_index}.npz").keys()]
            for array_key in array_keys:
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
                                    params.get("d_embed", params["d_model"]),
                                ),
                                dtype=jnp.float32,
                            )
                        )
                        tensor_index += 1

                    npz = np.load(f"{ckpt_dir}shard_{shard_index}/{file_index}.npz")
                    array = npz[array_key]
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
        if hf_conversion_callback.nested:
            return
        hf_conversion_callback.nested = True
        if utils.num_shards is None or utils.current_shard == 0:
            if utils.num_shards is not None:
                num_tensors = len(
                    utils.get_sharded_checkpoint_num_tensors(
                        utils.from_pretrained_model_name,
                        utils.from_pretrained_index_filename,
                        **utils.from_pretrained_kwargs,
                    )
                )
            else:
                num_tensors = len(model_dict)
            print(flush=True)
            utils.bar = tqdm(total=num_tensors, desc="Loading model tensors")
        with zipfile.ZipFile(f, "r") as z:
            try:
                last_storage_key = None
                f = None
                if utils.num_shards is not None:
                    utils.current_shard += 1
                for key in sorted(
                    model_dict.keys(),
                    key=lambda k: (model_dict[k].key, model_dict[k].seek_offset),
                ):
                    model_spec_key = max((k for k in model_spec.keys() if key.endswith(k)), key=len, default=None)

                    if model_spec_key is None:
                        model_dict[key] = torch.empty(
                            model_dict[key].shape,
                            dtype=model_dict[key].dtype,
                            device="meta",
                        )
                        utils.bar.update(1)
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
                    spec = model_spec[model_spec_key]
                    transforms = set(spec.get("transforms", ()))
                    tensor = model_dict[key].materialize(f, map_location="cpu")
                    model_dict[key] = tensor.to("meta")

                    if "remove_first_two_rows" in transforms:
                        tensor = tensor[2:]
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

                    utils.bar.update(1)

                if (
                    utils.num_shards is not None
                    and utils.current_shard < utils.num_shards
                ):
                    return

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
                if utils.num_shards is None or utils.current_shard >= utils.num_shards:
                    utils.bar.close()
                    utils.bar = None
                hf_conversion_callback.nested = False
                if isinstance(f, zipfile.ZipExtFile):
                    f.close()

    hf_conversion_callback.nested = False
    return hf_conversion_callback


@shatter("sb", "s")
def _init_opt_state(params, aux: jnp.DeviceArray):
    return mesh_transformer.util.to_f32(_init_opt_state.optimizer.init(params))


def init_opt_state(params, optimizer):
    """Returns initialized optax state for the given haiku parameters and optax optimizer."""
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


def initialize(
    quiet: Optional[bool] = None, driver_version="tpu_driver0.1_dev20210607"
):
    if quiet is None:
        quiet = exceptions.default_quiet
    global initialized, BACKEND
    if initialized:
        return
    if (
        "COLAB_TPU_ADDR" not in os.environ
        or len(os.environ["COLAB_TPU_ADDR"].strip()) == 0
    ) and ("TPU_NAME" not in os.environ or len(os.environ["TPU_NAME"].strip()) == 0):
        raise exceptions.ConfigurationError(
            "No Colab TPU detected.  Is this script running in a Colab TPU instance?",
            code=16,
            quiet=quiet,
        )
    print(
        termcolor.colored("\n\nConnecting to your Colab instance's TPU...", "magenta"),
        flush=True,
    )
    spinner = show_spinner()
    if "COLAB_TPU_ADDR" in os.environ and os.environ["COLAB_TPU_ADDR"].strip():
        tpu_addr = os.environ["COLAB_TPU_ADDR"]
        BACKEND = "colab"
    else:
        tpu_addr = os.environ["TPU_NAME"]
        BACKEND = "kaggle"
    tpu_addr = tpu_addr.replace("grpc://", "")
    tpu_addr_without_port = tpu_addr.split(":")[0]
    requests.post(
        f"http://{tpu_addr_without_port}:8475/requestversion/{driver_version}"
    )
    jax.config.FLAGS.jax_xla_backend = "tpu_driver"
    jax.config.FLAGS.jax_backend_target = "grpc://" + tpu_addr
    spinner.terminate()
    print(flush=True)
    if jax.device_count() < 8:
        raise exceptions.ConfigurationError(
            "We couldn't detect your Colab instance's TPU.\nTry restarting the runtime (Runtime > Restart Runtime) and trying again.",
            code=2,
            quiet=quiet,
        )
    initialized = True


def initialize_thread_resources(shards: int, backend=None):
    if not initialized:
        raise CoreNotInitializedError(
            "`initialize_thread_resources()` called before `initialize()` was called"
        )
    global thread_resources_initialized
    mesh_shape = (1, shards)
    devices = np.array(jax.devices(backend)[:shards]).reshape(mesh_shape)
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
    thread_resources_initialized = True
    return thread_resources_env


def get_tokenizer(
    params: dict, tokenizer_id: Optional[str] = None
) -> transformers.PreTrainedTokenizerBase:
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            params.get("tokenizer_id", "gpt2")
        )
    except ValueError:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            params.get("tokenizer_id", "gpt2"), use_fast=False
        )

    @contextlib.contextmanager
    def _mtj_softtuner_no_prefix():
        add_bos_token = getattr(tokenizer, "add_bos_token", False)
        add_prefix_space = getattr(tokenizer, "add_prefix_space", False)
        tokenizer.add_bos_token = False
        tokenizer.add_prefix_space = False
        try:
            yield
        finally:
            tokenizer.add_bos_token = add_bos_token
            tokenizer.add_prefix_space = add_prefix_space

    tokenizer._mtj_softtuner_no_prefix = _mtj_softtuner_no_prefix
    return tokenizer


def get_hf_checkpoint_metadata(ckpt_path: str):
    if os.path.exists(os.path.join(ckpt_path, "shard_0/0.npz")):
        return None

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
        if not isinstance(v, list):
            params[k] = params[v]
            continue
        for i, e in enumerate(v):
            if i == len(v) - 1:
                params[k] = e
            elif e in params:
                params[k] = params[e]
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
    params["max_batch_size"] = (
        450 if BACKEND == "colab" and params["d_model"] > 4096 else 2048
    )
    params["eos_token"] = (
        [50259, 50259] if model_config.model_type == "xglm" else [50256]
    )
    params["seq"] = 2048
    return lazy_load_spec, model_spec, params, tokenizer, newlinemode
