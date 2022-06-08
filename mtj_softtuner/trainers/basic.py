from .. import core
from .. import trainer_base

import os
import jax.numpy as jnp
import jax
import numpy as np
import transformers
from typing import List, Optional


class BasicTrainer(trainer_base.TrainerBase):
    class TrainerData(trainer_base.TrainerBase.TrainerData):
        def __init__(self):
            super().__init__()
            self.dataset_file: Optional[str] = None
            self.initial_softprompt: Optional[List[int]] = None

    data: "BasicTrainer.TrainerData"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset: Optional[np.array] = None

    def startup(self, step: int) -> None:
        if self.get_num_sequences() < self.data.gradient_accumulation_steps:
            self.raise_configuration_error(
                "Your dataset is too small!  gradient_accumulation_steps must be less than or equal to the number of sequences.",
                code=101,
            )
        if (
            self.data.kaiming_size <= 0
            and step < 0
            and self.data.initial_softprompt is None
        ):
            self.raise_configuration_error(
                "You have not set an initial soft prompt string.", code=103
            )
        if self.data.kaiming_size <= 0 and step < 0:
            self.data.soft_in_dim = len(self.data.initial_softprompt)

    def get_batch(self, step: int, size: int) -> np.ndarray:
        return self.dataset[(step - 1) * size : step * size]

    def get_num_sequences(self) -> int:
        if self.dataset is None:
            if self.data.dataset_file is None or not os.path.exists(
                self.data.dataset_file
            ):
                self.raise_configuration_error(
                    f"Dataset file not found at {repr(self.data.dataset_file)}",
                    code=102,
                )
            self.dataset = np.load(self.data.dataset_file, mmap_mode="r")
        assert self.dataset.ndim >= 2
        assert self.dataset.shape[0] >= 2
        return self.dataset.shape[0]

    def get_initial_soft_embeddings(
        self, network: core.EmbeddingCausalTransformer
    ) -> np.ndarray:
        if self.data.kaiming_size > 0:
            return jax.nn.initializers.he_normal()(
                jax.random.PRNGKey(1000000007),
                (
                    self.data.kaiming_size,
                    self.data.params.get("d_embed", self.data.params["d_model"]),
                ),
                dtype=jnp.float32,
            )
        return network.get_embedding_matrix(
            np.array(self.data.initial_softprompt, dtype=np.uint32)
        )

    def tokenize_dataset_callback(
        self, tokenizer: transformers.PreTrainedTokenizerBase, text: str
    ) -> List[int]:
        if self.data.newlinemode == "s":
            text = text.replace("\n", "</s>")
        return tokenizer.encode(text) + self.data.params["eos_token"]
