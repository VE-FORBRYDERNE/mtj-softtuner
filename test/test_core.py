import mtj_softtuner.core
import mtj_softtuner.testing_utils

import pytest
import jax.numpy as jnp
import jax
import torch
import torch.utils.dlpack

mtj_softtuner.core.initialized = mtj_softtuner.core.thread_resources_initialized = False


def shatter_sub():
    @mtj_softtuner.core.shatter("sb", "bs")
    def sample(s: jnp.DeviceArray, b: jnp.DeviceArray):
        return b - 1 - jax.lax.axis_index("batch"), jax.lax.psum(
            s, "shard"
        ) - 2 * jax.lax.axis_index("shard")

    actual = sample(jnp.arange(8, dtype=jnp.uint32), jnp.array([0], dtype=jnp.int32))
    expected = (
        jnp.array([-1], dtype=jnp.int32),
        28 - 2 * jnp.arange(8, dtype=jnp.int32),
    )
    assert isinstance(actual, tuple)
    assert len(actual) == 2
    assert all(isinstance(el, jnp.DeviceArray) for el in actual)
    assert all((a == e).all() and a.dtype is e.dtype for a, e in zip(actual, expected))


def get_hf_checkpoint_metadata_sub(p, h, n):
    model_spec = mtj_softtuner.core.get_hf_checkpoint_metadata(p)[1]
    assert any(k.startswith(h + str(n - 1)) for k in model_spec.keys())
    assert all(not k.startswith(h + str(n)) for k in model_spec.keys())


def reshard_reverse_sub(shape, old_shape, shards, negative=False, **k):
    assert (
        mtj_softtuner.core.reshard_reverse(
            torch.empty(shape, dtype=torch.float16),
            old_shape=old_shape,
            total_shards=shards,
            **k
        ).shape
        == old_shape
    ) != negative


def test_thread_resources_env_error():
    with pytest.raises(mtj_softtuner.core.CoreNotInitializedError):

        @mtj_softtuner.testing_utils.core_partly_initialized
        def f():
            pass

        f()


def test_shatter_error():
    with pytest.raises(mtj_softtuner.core.CoreNotInitializedError):
        shatter_sub()


@mtj_softtuner.testing_utils.core_dummy_initialized
def test_shatter_error_2():
    with pytest.raises(mtj_softtuner.core.CoreNotInitializedError):
        shatter_sub()


@mtj_softtuner.testing_utils.core_fully_initialized
def test_shatter():
    shatter_sub()


def decode(tokenizer):
    return (
        tokenizer.decode(
            [
                4342,
                788,
                318,
                530,
                886,
                25,
                257,
                2457,
                719,
                286,
                3555,
                11,
                257,
                2457,
                719,
                286,
                7327,
                13,
                50257,
            ]
        ),
        "Here then is one end: a final act of reading, a final act of consumption.",
    )


def test_get_tokenizer():
    actual, expected = decode(
        mtj_softtuner.core.get_tokenizer({"tokenizer_id": "gpt2"})
    )
    assert expected == actual


def test_get_tokenizer_2():
    actual, expected = decode(
        mtj_softtuner.core.get_tokenizer(
            {"tokenizer_id": "KoboldAI/fairseq-dense-125M"}
        )
    )
    assert expected + "<s>" == actual


def test_get_tokenizer_error():
    with pytest.raises(OSError):
        decode(mtj_softtuner.core.get_tokenizer({"tokenizer_id": "@"}))


def test_get_hf_checkpoint_metadata_neo():
    get_hf_checkpoint_metadata_sub("EleutherAI/gpt-neo-125M", "transformer.h.", 12)


def test_get_hf_checkpoint_metadata_rstrip():
    get_hf_checkpoint_metadata_sub("EleutherAI/gpt-neo-125M/", "transformer.h.", 12)


def test_get_hf_checkpoint_metadata_j():
    get_hf_checkpoint_metadata_sub("EleutherAI/gpt-j-6B", "transformer.h.", 28)


def test_get_hf_checkpoint_metadata_xglm():
    get_hf_checkpoint_metadata_sub("KoboldAI/fairseq-dense-125M", "model.layers.", 12)


def test_reshard_reverse():
    reshard_reverse_sub((1, 80, 40), (8, 10, 40), 8)


def test_reshard_reverse_2():
    reshard_reverse_sub((1, 17, 35), (7, 17, 5), 7)


def test_reshard_reverse_3():
    reshard_reverse_sub((1, 48), (8, 48), 8)


def test_reshard_reverse_4():
    reshard_reverse_sub((1, 42), (6, 7), 6)


def test_reshard_reverse_error():
    with pytest.raises(Exception):
        reshard_reverse_sub((1, 42), (6, 8), 6)


def test_reshard_reverse_error_2():
    with pytest.raises(Exception):
        reshard_reverse_sub((1, 80, 40), (8, 10, 5), 8)


def test_reshard_reverse_error_3():
    with pytest.raises(Exception):
        reshard_reverse_sub((1, 80, 40), (7, 10, 40), 8)


def test_reshard_reverse_error_4():
    with pytest.raises(Exception):
        reshard_reverse_sub((1, 80, 40, 2), (8, 10, 40, 2), 8)


@mtj_softtuner.testing_utils.core_fully_initialized
def test_dlpack():
    @mtj_softtuner.core.shatter("sb", "b")
    def f(_, v):
        return v + 1

    assert (
        f(
            jnp.empty(8),
            mtj_softtuner.core.jax_from_dlpack(
                torch.utils.dlpack.to_dlpack(
                    torch.eye(2056, dtype=torch.float32)[1:].T[None]
                )
            ),
        )
        == jnp.array(torch.eye(2056, dtype=torch.float32)[1:].T[None] + 1)
    ).all()
