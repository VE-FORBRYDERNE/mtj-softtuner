import mtj_softtuner
import mtj_softtuner.testing_utils
import os
import requests
import jax.numpy as jnp
import numpy as np
import torch
import pytest


def basic_trainer_sub(ckpt_path, save_file, prompt_method="tokens", soft_in_dim=51):
    """
    This test just makes sure that the BasicTrainer example can be run on a
    CPU without crashing or raising an error.
    """
    if os.path.isfile(save_file):
        os.remove(save_file)
    data = requests.get(
        "https://archive.org/download/TheLibraryOfBabel/babel_djvu.txt"
    ).content.decode("utf8")
    with open("dataset.txt", "w") as f:
        f.write(data)
    trainer = mtj_softtuner.BasicTrainer()
    trainer.data.ckpt_path = ckpt_path
    trainer.set_params("GPT-Neo-125M")
    trainer.get_hf_checkpoint_metadata()
    trainer.data.save_file = save_file
    trainer.data.prompt_method = prompt_method
    if prompt_method == "tokens":
        initial_softprompt = (
            "Le Jeu du Prochain Train itself is simplicity in motion. The object: "
            "Be the last of your round's six to jump from one side of the tracks to "
            "the other - that is, across the tracks - before the train passes.\n\n"
        )
        tokenizer = trainer.get_tokenizer()
        if trainer.data.newlinemode == "s":
            initial_softprompt = initial_softprompt.replace("\n", "</s>")
        trainer.data.initial_softprompt = tokenizer.encode(
            initial_softprompt, max_length=int(2e9), truncation=True
        )
    else:
        trainer.data.soft_in_dim = soft_in_dim
    dataset_path = "dataset.txt"
    output_file = "dataset.npy"
    batch_size = 128
    epochs = 3
    trainer.tokenize_dataset(dataset_path, output_file, batch_size, epochs)
    dataset_file = output_file
    trainer.data.dataset_file = dataset_file
    trainer.data.gradient_accumulation_steps = 3
    trainer.data.stparams = {
        "lr": 4.5e-5,
        "max_grad_norm": 10.0,
        "weight_decay": 0.1,
        "warmup": 0.1,
        "end_lr_multiplier": 0.1,
        "save_every": 5,
    }
    trainer.data.params["cores_per_replica"] = 1
    trainer.train(
        skip_initialize_thread_resources=True,
        skip_get_hf_checkpoint_metadata=True,
        hide_compiling_spinner=True,
    )
    output_file = "my_softprompt.zip"
    name = "Untitled"
    author = ""
    supported = "Generic 6B"
    description = "Baby shoes"
    trainer.export_to_kobold(output_file, name, author, supported, description)
    output_file = "my_softprompt.json"
    soft_prompt_name = "Untitled"
    soft_prompt_description = "Baby shoes"
    trainer.export_to_mkultra(output_file, soft_prompt_name, soft_prompt_description)


@pytest.mark.order(after="test_basic_trainer_fairseq")
@mtj_softtuner.testing_utils.core_fully_initialized
def test_basic_trainer_fairseq_vocab_sample():
    basic_trainer_sub(
        "KoboldAI/fairseq-dense-125M",
        "my_softprompt_vocab_sample.mtjsp",
        prompt_method="vocab_sample",
    )


@pytest.mark.order(after="test_basic_trainer_fairseq")
@mtj_softtuner.testing_utils.core_fully_initialized
def test_basic_trainer_fairseq_kaiming():
    basic_trainer_sub(
        "KoboldAI/fairseq-dense-125M",
        "my_softprompt_kaiming.mtjsp",
        prompt_method="kaiming",
    )


@mtj_softtuner.testing_utils.core_fully_initialized
def test_basic_trainer_fairseq():
    basic_trainer_sub("KoboldAI/fairseq-dense-125M", "my_softprompt.mtjsp")


@mtj_softtuner.testing_utils.core_fully_initialized
def test_basic_trainer_neo():
    basic_trainer_sub("EleutherAI/gpt-neo-125M", "my_softprompt_neo_hf.mtjsp")


pieces = 16
total_shards = 1
layers = 12
url = "https://huggingface.co/EleutherAI/gpt-neo-125M/resolve/main/pytorch_model.bin"


def get_old_shape(t, dim=2):
    if len(t.shape) == 2:
        shard_shape = t.shape
        if dim == 1:
            assert shard_shape[0] % total_shards == 0
            return (shard_shape[0] // total_shards, shard_shape[1])
        elif dim == 2:
            assert shard_shape[1] % total_shards == 0
            return (shard_shape[0], shard_shape[1] // total_shards)
        else:
            raise ValueError(f"unsupported dim {dim}")
    if len(t.shape) == 1:
        assert t.shape[0] % total_shards == 0
        return (t.shape[0] // total_shards,)
    else:
        raise ValueError(f"unsupported shape {t.shape}")


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def save(cpu_flattened):
    for i in range(total_shards):
        cpu_flattened_chunked = split(cpu_flattened, pieces)
        for j, chunk in enumerate(cpu_flattened_chunked):
            with open(f"jax_checkpoint/shard_{i}/{j}.npz", "wb") as f:
                np.savez(f, *map(lambda c: c[i], chunk))


transforms = [
    ("transformer.wpe.weight", False, 2),
    ("transformer.wte.weight", False, 1),
]

layer_names = sorted(map(str, range(layers)))
for layer in layer_names:
    transforms.extend(
        [
            (f"transformer.h.{layer}.attn.attention.q_proj.weight", False, 2),
            (f"transformer.h.{layer}.attn.attention.v_proj.weight", False, 2),
            (f"transformer.h.{layer}.attn.attention.k_proj.weight", False, 2),
            (f"transformer.h.{layer}.attn.attention.out_proj.bias", True, None),
            (f"transformer.h.{layer}.attn.attention.out_proj.weight", False, 1),
            (f"transformer.h.{layer}.mlp.c_fc.bias", False, 1),
            (f"transformer.h.{layer}.mlp.c_fc.weight", False, 2),
            (f"transformer.h.{layer}.mlp.c_proj.bias", True, None),
            (f"transformer.h.{layer}.mlp.c_proj.weight", False, 1),
            (f"transformer.h.{layer}.ln_1.bias", False, None),
            (f"transformer.h.{layer}.ln_1.weight", False, None),
            (f"transformer.h.{layer}.ln_2.bias", False, None),
            (f"transformer.h.{layer}.ln_2.weight", False, None),
        ]
    )
transforms.extend(
    [
        ("transformer.ln_f.bias", False, None),
        ("transformer.ln_f.weight", False, None),
    ]
)


@mtj_softtuner.testing_utils.core_fully_initialized
def test_read_ckpt_neo():
    for i in range(total_shards):
        os.makedirs(os.path.join("jax_checkpoint", f"shard_{i}"), exist_ok=True)

    checkpoint = []

    data = requests.get(url).content
    with open("checkpoint.pt", "wb") as f:
        f.write(data)
    torch_checkpoint = torch.load("checkpoint.pt", map_location="cpu")

    for transform in transforms:
        params = torch_checkpoint[transform[0]]

        if transform[0] in ("transformer.wte.weight", "lm_head.weight"):
            params = torch.cat((params, torch.zeros(143, params.shape[1])))

        if not any(s in transform[0] for s in ("wte", "wpe")) and params.ndim == 2:
            params = params.T

        if transform[2] is not None:
            old_shape = (total_shards,) + get_old_shape(params, transform[2])
        else:
            old_shape = (
                total_shards,
                params.shape[0],
            )

        params = np.asarray(params[None], dtype=jnp.bfloat16)

        assert params.shape == old_shape
        checkpoint.append(params)

    checkpoint.append(np.zeros(total_shards, dtype=np.int32))
    save(checkpoint)

    basic_trainer_sub("jax_checkpoint", "my_softprompt_neo_jax.mtjsp")


@pytest.mark.order("last")
def test_basic_sp_equivalence_neo():
    """
    This test makes sure that the outputs of the two tests that use
    GPT-Neo models output the same soft prompt tensor
    """
    sp_hf = np.load("my_softprompt_neo_hf.mtjsp")["tensor"]
    sp_jax = np.load("my_softprompt_neo_jax.mtjsp")["tensor"]
    assert all(isinstance(t, np.ndarray) for t in (sp_hf, sp_jax))
    assert sp_hf.dtype is sp_jax.dtype
    assert sp_hf.shape == sp_jax.shape
    assert np.allclose(sp_hf, sp_jax, rtol=1e-5, atol=1e-6)
