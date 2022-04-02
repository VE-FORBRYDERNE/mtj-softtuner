from . import _patch as _
from . import core
from . import exceptions
from . import serialization
from . import visualization
from .kobold import torch_lazy_loader

import abc
import os
import termcolor
from tqdm.auto import tqdm
from typing import Optional
import math
import jax
import jax.numpy as jnp
import jax.dlpack
import numpy as np
import haiku as hk
import optax
import ftfy
import zipfile
import json
import torch
import base64
import pickle
import datetime
import uuid
from typing import List, TextIO, Union
import mesh_transformer
import mesh_transformer.util
import mesh_transformer.layers
import mesh_transformer.transformer_shard
import transformers


class TrainerBase(abc.ABC):
    @abc.abstractmethod
    def startup(self, step: int) -> None:
        ...

    @abc.abstractmethod
    def get_batch(self, step: int, size: int) -> np.array:
        ...

    @abc.abstractmethod
    def get_num_sequences(self) -> int:
        ...

    @abc.abstractmethod
    def get_initial_soft_embeddings(
        self, network: core.EmbeddingCausalTransformer
    ) -> np.array:
        ...

    @abc.abstractmethod
    def tokenize_dataset_callback(
        self, tokenizer: transformers.PreTrainedTokenizerBase, text: str
    ) -> List[int]:
        ...

    class TrainerData:
        def __init__(self):
            self.lazy_load_spec: Optional[dict] = None
            self.model_spec: Optional[dict] = None
            self.tokenizer_id: Optional[str] = None
            self.newlinemode: Optional[str] = None
            self.ckpt_path: Optional[str] = None
            self.save_file: Optional[str] = None
            self.params: Optional[dict] = None
            self.stparams: Optional[dict] = None
            self.gradient_accumulation_steps = -1
            self.soft_in_dim = -1

    data: TrainerData

    def __init__(self, universe: Optional[int] = None, quiet=False):
        self.quiet = quiet
        self.universe = universe
        self.data = self.TrainerData()
        self._spmodule: Optional[str] = None
        if universe is not None:
            try:
                self.data = serialization.restore_variable(
                    universe, type(self).__name__ + "_" + "data"
                )
            except ValueError:
                pass

    def raise_configuration_error(self, msg, **kwargs):
        if "quiet" not in kwargs:
            kwargs["quiet"] = self.quiet
        raise exceptions.ConfigurationError(msg, **kwargs)

    def save_data(self):
        if self.data.params is not None:
            for p in ("optimizer", "soft_in_dim"):
                self.data.params.pop(p, None)
        serialization.save_variable(
            self.universe,
            type(self).__name__ + "_" + "data",
            self.data,
        )

    def get_hf_checkpoint_metadata(self) -> bool:
        data = core.get_hf_checkpoint_metadata(self.data.ckpt_path)
        if data is None:
            return False
        (
            self.data.lazy_load_spec,
            self.data.model_spec,
            self.data.params,
            self.data.tokenizer_id,
            self.data.newlinemode,
        ) = data
        return True

    def get_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        self.get_hf_checkpoint_metadata()
        return core.get_tokenizer(self.data.params)

    def set_params(self, model_type: str):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "model_params.json"
        )
        with open(path) as f:
            param_map: dict = json.load(f)
        if model_type not in param_map:
            self.raise_configuration_error(
                f"Unknown model type {repr(model_type)}", code=15
            )
        spec = param_map[model_type]
        self.data.tokenizer_id = spec["tokenizer_id"]
        self.data.newlinemode = spec["newlinemode"]
        self.data.params = spec["params"]

    def export_to_kobold(
        self, output_file: str, name: str, author: str, supported: str, description: str
    ):
        try:
            npz = np.load(self.data.save_file, allow_pickle=True)
            assert npz["step"] > 0
            assert npz["tensor"].ndim == 2 and "opt_state" in npz
            assert npz["tensor"].shape[0] < self.data.params["max_batch_size"]
            assert npz["tensor"].shape[1] == self.data.params["d_model"]
            assert all(
                p in npz
                for p in (
                    "loss",
                    "last_loss",
                    "grad_norm",
                    "grad_norm_micro",
                )
            )
            _step = np.uint32(npz["step"]).item()
        except AssertionError:
            self.raise_configuration_error("MTJSP file is corrupted.", code=14)

        tensor = npz["tensor"]
        if tensor.dtype == "V2":
            tensor.dtype = jnp.bfloat16

        meta = {
            "name": name,
            "author": author,
            "supported": supported,
            "description": description,
        }
        if len(meta["author"].strip()) == 0:
            meta.pop("author")
        meta["supported"] = list(map(lambda m: m.strip(), supported.split(",")))

        with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_LZMA) as z:
            with z.open("tensor.npy", "w") as f:
                np.save(f, tensor, allow_pickle=False)
        with zipfile.ZipFile(output_file, "a", compression=zipfile.ZIP_STORED) as z:
            with z.open("meta.json", "w") as f:
                f.write(json.dumps(meta, indent=2).encode("utf-8"))

    def export_to_mkultra(
        self, output_file: str, soft_prompt_name: str, soft_prompt_description: str
    ):
        try:
            npz = np.load(self.data.save_file, allow_pickle=True)
            assert npz["step"] > 0
            assert npz["tensor"].ndim == 2 and "opt_state" in npz
            assert npz["tensor"].shape[0] < self.data.params["max_batch_size"]
            assert npz["tensor"].shape[1] == self.data.params["d_model"]
            assert all(
                p in npz
                for p in (
                    "loss",
                    "last_loss",
                    "grad_norm",
                    "grad_norm_micro",
                )
            )
            _step = np.uint32(npz["step"]).item()
        except AssertionError:
            self.raise_configuration_error("MTJSP file is corrupted.", code=14)

        tensor = npz["tensor"]
        if tensor.dtype == "V2":
            tensor.dtype = jnp.bfloat16
            tensor = torch.tensor(tensor).to(torch.float32)
        else:
            tensor = torch.tensor(tensor)

        with open(output_file, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "step": _step,
                        "loss": float(npz["loss"].item()),
                        "uuid": str(uuid.uuid4()),
                        "name": soft_prompt_name,
                        "description": soft_prompt_description,
                        "epoch": datetime.datetime.now().timestamp(),
                    },
                    "tensor": base64.b64encode(
                        pickle.dumps(
                            tensor,
                            protocol=4,
                        ),
                    ).decode("ascii"),
                },
                f,
            )

    def tokenize_dataset(
        self,
        dataset_path: Union[str, TextIO],
        output_file: Union[str, TextIO],
        batch_size=2048,
        epochs=1,
        use_ftfy=True,
    ):
        dataset_path = dataset_path.replace("\\", "/")
        output_file = output_file.replace("\\", "/")
        if not isinstance(batch_size, int) or batch_size < 1:
            self.raise_configuration_error(
                "batch_size must be an integer greater than zero.", code=9
            )
        if not isinstance(epochs, int) or epochs < 1:
            self.raise_configuration_error(
                "epochs must be an integer greater than zero.", code=10
            )
        if isinstance(output_file, str) and output_file.endswith("/"):
            self.raise_configuration_error(
                "output_file should be the path to a file, not a directory.", code=11
            )
        if isinstance(dataset_path, str) and not os.path.exists(dataset_path):
            self.raise_configuration_error(
                "dataset_path is not set to a valid file or directory.", code=12
            )

        tokenizer = self.get_tokenizer()

        batch_size = min(
            batch_size,
            self.data.params["max_batch_size"] - self.data.soft_in_dim,
        )
        assert batch_size >= 0
        print(
            termcolor.colored(
                "\nIf you see a warning somewhere below about token indices, ignore it.  That warning is normal.\n",
                "magenta",
            )
        )
        print("Batch size:", batch_size)
        print(termcolor.colored("Tokenizing your dataset...\n", "magenta"))

        if not isinstance(dataset_path, str):
            files = [dataset_path]
        elif os.path.isfile(dataset_path):
            files = [dataset_path]
        else:
            files = (
                os.path.join(dataset_path, filename)
                for filename in os.listdir(dataset_path)
            )
        tokens = []
        eos = tokenizer.decode(self.data.params["eos_token"])
        for path in files:
            if isinstance(path, str):
                f = open(path)
            else:
                f = path
            try:
                text = f.read()
                if use_ftfy:
                    text = ftfy.fix_text(text)
                text = text.replace("<|endoftext|>", eos)
                tokens.extend(self.tokenize_dataset_callback(tokenizer, text))
            finally:
                if isinstance(path, str):
                    f.close()

        print("Dataset size (in tokens):", len(tokens))
        if len(tokens) < batch_size + 1:
            self.raise_configuration_error(
                "Your dataset is too small!  The number of tokens has to be greater than the batch size.",
                code=13,
            )
        tail = len(tokens) % (batch_size + 1)
        if tail:
            print(
                f"We're removing the last {tail} tokens from your dataset to make the length a multiple of {batch_size+1}."
            )
            tokens = tokens[:-tail]

        tokens = np.array(tokens, dtype=np.uint16).reshape((-1, batch_size + 1))
        if epochs > 1:
            rng = np.random.Generator(np.random.PCG64(1729))
            tokens = np.concatenate(
                (
                    tokens,
                    *(rng.permutation(tokens, axis=0) for i in range(epochs - 1)),
                ),
                axis=0,
            )
        print(f"Total sequences in your dataset: {tokens.shape[0]}")

        if isinstance(output_file, str):
            f = open(output_file, "w")
        else:
            f = output_file
        try:
            np.save(output_file, tokens)
        finally:
            if isinstance(output_file, str):
                f.close()

        self.save_data()

    def train(self):
        if self.data.ckpt_path is None:
            self.raise_configuration_error(
                "You didn't specify the path to your model.", code=3
            )
        elif self.data.save_file is None:
            self.raise_configuration_error(
                "You did not set the path for the save file.", code=4
            )
        elif self.data.stparams is None:
            self.raise_configuration_error(
                "You have not set soft-tuning hyperparameters.", code=7
            )
        elif self.data.gradient_accumulation_steps < 0:
            self.raise_configuration_error(
                "You have not set gradient accumulation steps.", code=8
            )

        self.save_data()

        if (
            not os.path.exists(os.path.join(self.data.ckpt_path, "shard_0/0.npz"))
            and not self.get_hf_checkpoint_metadata()
        ):
            raise RuntimeError("Error getting HF checkpoint metadata")

        core.initialize(quiet=self.quiet)
        core.initialize_thread_resources(self.data.params["cores_per_replica"])

        step = 0

        # Set up the scheduler which determines the learning rate for each step
        steps = self.get_num_sequences() // self.data.gradient_accumulation_steps
        warmup_steps = max(1, round(steps * self.data.stparams["warmup"]))
        scheduler = mesh_transformer.util.gpt3_schedule(
            warmup_steps,
            max(1, steps - warmup_steps),
            self.data.stparams["lr"],
            self.data.stparams["end_lr_multiplier"] * self.data.stparams["lr"],
        )

        # Tell Mesh Transformer to create the network as bfloat16
        self.data.params["early_cast"] = True

        g_avg = s_avg = False
        if not os.path.exists(self.data.save_file):
            print("We are starting a brand new soft-tuning session.\n")
            self.startup(step=-1)
            if self.data.soft_in_dim <= 0:
                self.raise_configuration_error(
                    "You have not set a soft prompt size.", code=6
                )
            g_avg = s_avg = True
        else:
            # If we're resuming a soft-tuning session, the soft prompt tensor is
            # already in the save file and we just have to decode it.
            try:
                npz = np.load(self.data.save_file, allow_pickle=True)
                assert npz["step"] > 0
                assert npz["tensor"].ndim == 2 and "opt_state" in npz
                assert npz["tensor"].shape[0] < self.data.params["max_batch_size"]
                assert npz["tensor"].shape[1] == self.data.params["d_model"]
                assert all(
                    p in npz
                    for p in (
                        "loss",
                        "last_loss",
                        "grad_norm",
                        "grad_norm_micro",
                    )
                )
                self.data.soft_in_dim = npz["tensor"].shape[0]
                step = np.uint32(npz["step"]).item()
                if "g_avg" in npz and "s_avg" in npz:
                    g_avg = np.float32(npz["g_avg"])
                    s_avg = np.float32(npz["s_avg"])
            except AssertionError:
                raise exceptions.ConfigurationError("MTJSP file is corrupted.", code=14)
            print(f"We're resuming a previous soft-tuning session at step {step+1}.\n")
            self.startup(step=step + 1)
            soft_embeddings = npz["tensor"]
            if soft_embeddings.dtype == "V2":
                soft_embeddings.dtype = jnp.bfloat16
            soft_embeddings = jnp.float32(soft_embeddings)

        # Load the model
        self.data.params["soft_in_dim"] = self.data.soft_in_dim
        if self._spmodule is None:
            print(termcolor.colored("Initializing network...", "magenta"), flush=True)
            self.data.params["optimizer"] = optax.scale(0)
            network = core.EmbeddingCausalTransformer(
                self.data.params,
                **(
                    {"dematerialized": True}
                    if core.DEMATERIALIZED_LOADING_SUPPORTED
                    else {}
                ),
            )
            shards_in = self.data.params["cores_per_replica"]
            if core.DEMATERIALIZED_LOADING_SUPPORTED:
                network.state["params"][
                    "causal_transformer_shard/~/embedding_shard/~/softtune_linear"
                ] = {
                    "w": mesh_transformer.transformer_shard.PlaceholderTensor(
                        shards_in,
                        math.ceil(self.data.soft_in_dim / shards_in),
                        self.data.params["d_model"],
                    )
                }
            print(
                termcolor.colored("\n\nLoading pretrained model...", "magenta"),
                flush=True,
            )
            if os.path.exists(os.path.join(self.data.ckpt_path, "shard_0/0.npz")):
                if self.data.params is None:
                    self.raise_configuration_error(
                        "You have not specified the type of model you are going to train.",
                        code=9,
                    )
                network.state, self._spmodule = core.read_ckpt_custom(
                    self.data.soft_in_dim,
                    network.move_xmap,
                    network.config,
                    network.state,
                    self.data.ckpt_path,
                    self.data.params["cores_per_replica"],
                )
            else:
                self._spmodule = (
                    "causal_transformer_shard/~/embedding_shard/~/softtune_linear"
                )
                network.state["params"][self._spmodule]["w"] = jnp.empty(
                    (
                        shards_in,
                        math.ceil(self.data.soft_in_dim / shards_in),
                        self.data.params["d_model"],
                    ),
                    dtype=jnp.float32,
                )
                with torch_lazy_loader.use_lazy_torch_load(
                    callback=core.get_hf_conversion_callback(
                        network, self.data.model_spec
                    ),
                    dematerialized_modules=True,
                ):
                    transformers.AutoModelForCausalLM.from_pretrained(
                        self.data.ckpt_path
                    )
            network.state = network.move_xmap(
                network.state, np.zeros(self.data.params["cores_per_replica"])
            )
            network.state["params"][self._spmodule]["w"] = jnp.float32(
                network.state["params"][self._spmodule]["w"]
            )

        # Set up the optimizer, which is the algorithm we use to train the soft prompt
        network.config["optimizer"] = optax.chain(
            optax.scale(1 / self.data.gradient_accumulation_steps),
            mesh_transformer.util.clip_by_global_norm(
                float(self.data.stparams["max_grad_norm"])
            ),
            optax.scale_by_adam(mu_dtype=jnp.float32),
            mesh_transformer.util.additive_weight_decay(
                self.data.stparams["weight_decay"]
            ),
            optax.scale(-1),
            optax.scale_by_schedule(scheduler),
        )

        if step == 0:
            soft_embeddings = jnp.float32(self.get_initial_soft_embeddings(network))
            network.state["opt_state"] = core.init_opt_state(
                network.state["params"][self._spmodule],
                network.config["optimizer"],
            )
        else:
            # Optimizer state is already saved otherwise
            network.state["opt_state"] = mesh_transformer.util.to_f32(
                tuple(npz["opt_state"])
            )

        # Pad the embedding matrix with zeros at the bottom so that its number of
        # rows is a multiple of 8 (or 4 for GPT-Neo-2.7B)
        rows = soft_embeddings.shape[0]
        padding_amount = -(rows % -self.data.params["cores_per_replica"])
        soft_embeddings = jnp.pad(soft_embeddings, ((0, padding_amount), (0, 0)))
        # Split the matrix row-wise into 8 (or 4) submatrices (so that if the original
        # matrix had R rows and C columns, then each submatrix has R/8 rows and C
        # columns) and then concatenate the 8 submatrices together along a new
        # leading axis into a 3-dimensional array so that it can be sharded by
        # xmapped functions
        soft_embeddings = soft_embeddings.reshape(
            (
                self.data.params["cores_per_replica"],
                -1,
                self.data.params["d_model"],
            )
        )
        # Put this 3D array into the network so we can train it
        network.state["params"][self._spmodule]["w"] = soft_embeddings

        def train_grad(state, ctx, tgt):
            @hk.without_apply_rng
            @hk.transform
            def inner(ctx, tgt):
                transformer = mesh_transformer.transformer_shard.CausalTransformerShard(
                    network.config
                )
                out = transformer.loss(ctx, tgt, z_loss=True)
                return out["loss"], out["last_loss"]

            def inner_wrapped(sp_params, params, *args, **kwargs):
                params[self._spmodule] = sp_params
                return inner.apply(params, *args, **kwargs)

            # Compute gradient and also the actual loss value
            params = mesh_transformer.util.to_bf16(state["params"])
            val_grad_fn = jax.value_and_grad(inner_wrapped, has_aux=True)
            (loss, last_loss), grad = val_grad_fn(
                params[self._spmodule], params, ctx, tgt
            )
            # Calculate the Euclidean norm of the modified gradient
            gnorm = mesh_transformer.util.global_norm(grad)
            # Return the modified gradient, the loss and last loss, and the
            # norm of the modified gradient
            return grad, loss, last_loss, gnorm

        @core.shatter("sbb", "bbbb")
        def train_initial(state, ctx, tgt):
            grad, loss, last_loss, gnorm = train_grad(state, ctx, tgt)
            return grad, loss, last_loss, gnorm

        @core.shatter("sbb", "b")
        def train_add_grads(_, grad1, grad2):
            return jax.tree_multimap(lambda a, b: a + b, grad1, grad2)

        @core.shatter("sbb", "bbs")
        def train_final(state, grad, gnorm):
            grad_norm_micro = jax.lax.pmean(gnorm, "batch")
            grad = jax.lax.pmean(grad, "batch")
            grad_norm = mesh_transformer.util.global_norm(grad)
            # Apply Adam algorithm to grad to get updates (output of the Adam
            # algorithm) and new_opt_state (new state of the Adam optimizer)
            updates, new_opt_state = network.config["optimizer"].update(
                grad, state["opt_state"], params=state["params"][self._spmodule]
            )
            # optax.apply_updates here just returns the element-wise sum
            # of state["params"][self._spmodule] and updates, cast to bfloat16.
            state["params"][self._spmodule] = optax.apply_updates(
                state["params"][self._spmodule], mesh_transformer.util.to_f32(updates)
            )
            return (
                grad_norm / self.data.gradient_accumulation_steps,
                grad_norm_micro,
                {
                    "params": state["params"],
                    "step": state["step"] + 1,
                    "opt_state": new_opt_state,
                },
            )

        def save_mtjsp(
            loss,
            last_loss,
            grad_norm,
            grad_norm_micro,
        ):
            tensor = network.state["params"][self._spmodule]["w"]
            tensor = tensor.reshape((-1, tensor.shape[2]))
            tensor = tensor[: self.data.soft_in_dim]
            with open(self.data.save_file, "wb") as f:
                np.savez_compressed(
                    f,
                    tensor=tensor,
                    opt_state=np.array(network.state["opt_state"], dtype=object),
                    step=np.uint32(step),
                    loss=np.float32(loss),
                    last_loss=np.float32(last_loss),
                    grad_norm=np.float32(grad_norm),
                    grad_norm_micro=np.float32(grad_norm_micro),
                    **(
                        {"g_avg": g_avg, "s_avg": s_avg}
                        if type(g_avg) is not bool
                        else {}
                    ),
                )
            self.save_data()

        def train_step(use_tqdm=True):
            # Get the next batch from the dataset
            data = self.get_batch(step, self.data.gradient_accumulation_steps)
            # Concatenate the soft prompt at the beginning
            vocab_size = self.data.params["n_vocab"] + self.data.params.get(
                "n_vocab_padding", 0
            )
            header = np.tile(
                np.arange(
                    vocab_size,
                    vocab_size + self.data.soft_in_dim,
                    dtype=np.uint32,
                ),
                (data.shape[0], 1),
            )
            data = np.concatenate((header, data), axis=1)[
                :, : self.data.params["max_batch_size"] + 1
            ]

            ctx = data[:, :-1]
            tgt = data[:, 1:]

            grad, loss, last_loss, gnorm = train_initial(
                network.state,
                ctx[np.newaxis, 0],
                tgt[np.newaxis, 0],
            )
            r = range(1, ctx.shape[0])
            if use_tqdm:
                r = tqdm(
                    r,
                    initial=1,
                    total=ctx.shape[0],
                    desc="GRADIENT ACCUMULATION",
                    leave=False,
                )
            for i in r:
                _grad, _loss, _last_loss, _gnorm = train_initial(
                    network.state,
                    ctx[np.newaxis, i],
                    tgt[np.newaxis, i],
                )
                grad = train_add_grads(
                    np.empty(self.data.params["cores_per_replica"]),
                    grad,
                    _grad,
                )
                loss += _loss
                last_loss += _last_loss
                gnorm += _gnorm
            loss /= ctx.shape[0]
            last_loss /= ctx.shape[0]
            gnorm /= ctx.shape[0]
            grad_norm, grad_norm_micro, network.state = train_final(
                network.state,
                grad,
                gnorm,
            )
            del grad

            return (
                np.array(loss).mean(),
                np.array(last_loss).mean(),
                np.array(grad_norm).mean(),
                np.array(grad_norm_micro).mean(),
            )

        def compute_noise(g_avg, s_avg):
            noise_scale_alpha = 0.01
            gbsmall = grad_norm_micro ** 2
            gbbig = grad_norm ** 2
            g = (self.data.gradient_accumulation_steps * gbbig - gbsmall) / (
                self.data.gradient_accumulation_steps - 1
            )
            s = (gbsmall - gbbig) / (1 - 1 / self.data.gradient_accumulation_steps)
            use_step_in_noise_avgs = gbbig < 2
            if use_step_in_noise_avgs:
                if g_avg is True:
                    g_avg = np.float32(g)
                elif g_avg is not False:
                    g_avg = (1 - noise_scale_alpha) * g_avg + noise_scale_alpha * g
                if s_avg is True:
                    s_avg = np.float32(s)
                elif s_avg is not False:
                    s_avg = (1 - noise_scale_alpha) * s_avg + noise_scale_alpha * s
                if type(g_avg) is not bool and type(s_avg) is not bool:
                    return s_avg / g_avg, g_avg, s_avg
            return None, g_avg, s_avg

        step += 1

        # Train
        first_step = step
        if step <= steps:
            print(
                termcolor.colored(
                    "Compiling trainer; this usually takes less than 6 minutes\n",
                    "magenta",
                ),
                flush=True,
            )
            # Simultaneously compile the trainer and train for one step
            spinner = core.show_spinner()
            loss, last_loss, grad_norm, grad_norm_micro = train_step(use_tqdm=False)
            spinner.terminate()
            # Show the plots for learning rate, etc.
            visualization.show_plots()
            # Update plot
            visualization.push_data(
                step,
                scheduler(step),
                loss,
                last_loss,
                grad_norm,
                grad_norm_micro,
            )
            if g_avg is not False:
                b_simple, g_avg, s_avg = compute_noise(g_avg, s_avg)
                visualization.push_noise_data(
                    step,
                    b_simple,
                    g_avg,
                    s_avg,
                )
        # Create a save file for step 1
        if step == 1 or step % self.data.stparams["save_every"] == 0:
            save_mtjsp(
                loss,
                last_loss,
                grad_norm,
                grad_norm_micro,
            )
        for _ in tqdm(
            range(first_step, steps),
            initial=first_step,
            total=steps,
            desc="SOFT-TUNING PROGRESS",
        ):
            step += 1
            # Train for one step and update the plot
            loss, last_loss, grad_norm, grad_norm_micro = train_step(use_tqdm=True)
            visualization.push_data(
                step,
                scheduler(step),
                loss,
                last_loss,
                grad_norm,
                grad_norm_micro,
            )
            if g_avg is not False:
                b_simple, g_avg, s_avg = compute_noise(g_avg, s_avg)
                visualization.push_noise_data(
                    step,
                    b_simple,
                    g_avg,
                    s_avg,
                )
            # Save whenever step is divisible by save_every
            if step % self.data.stparams["save_every"] == 0:
                save_mtjsp(
                    loss,
                    last_loss,
                    grad_norm,
                    grad_norm_micro,
                )
        step += 1
        save_mtjsp(
            loss,
            last_loss,
            grad_norm,
            grad_norm_micro,
        )
