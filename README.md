# mtj-softtuner [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VE-FORBRYDERNE/mtj-softtuner/blob/main/mtj-softtuner.ipynb)

[![Python package](https://github.com/VE-FORBRYDERNE/mtj-softtuner/actions/workflows/python-package.yml/badge.svg)](https://github.com/VE-FORBRYDERNE/mtj-softtuner/actions/workflows/python-package.yml) [![GitHub license](https://img.shields.io/github/license/VE-FORBRYDERNE/mtj-softtuner?color=informational)](https://github.com/VE-FORBRYDERNE/mtj-softtuner/blob/main/LICENSE) [![GitHub release (latest by date)](https://img.shields.io/github/v/release/VE-FORBRYDERNE/mtj-softtuner)](https://github.com/VE-FORBRYDERNE/mtj-softtuner/releases) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/5d95207f6e784dc2b56490c7bd8bb439)](https://www.codacy.com/gh/VE-FORBRYDERNE/mtj-softtuner/dashboard?utm_source=github.com&utm_medium=referral&utm_content=VE-FORBRYDERNE/mtj-softtuner&utm_campaign=Badge_Coverage) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/5d95207f6e784dc2b56490c7bd8bb439)](https://www.codacy.com/gh/VE-FORBRYDERNE/mtj-softtuner/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=VE-FORBRYDERNE/mtj-softtuner&amp;utm_campaign=Badge_Grade) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### (Unofficial Mesh Transformer JAX soft-tuning notebook)

Create, in Colab, soft prompts compatible with [KoboldAI](https://github.com/KoboldAI/KoboldAI-Client) and [mkultra](https://github.com/corolla-johnson/mkultra) for your favourite GPT-J-6B-based or GPT-Neo-2.7B-based model!

See this paper https://arxiv.org/pdf/2104.08691.pdf for more information about what a soft prompt is.

## Quickstart

If you're not a programmer or you want a demo of how to use the API, [click here to open the demo notebook.](https://colab.research.google.com/github/VE-FORBRYDERNE/mtj-softtuner/blob/main/mtj-softtuner.ipynb)

## API Usage

To install mtj-softtuner in a TPU Colab notebook, run these commands:

```bash
git clone https://github.com/ve-forbryderne/mtj-softtuner
bash mtj-softtuner/install.sh
```

Here's an extremely basic example of how to use the API:

```python
from mtj_softtuner import BasicTrainer

# Change this to an integer (e.g. 1) if you want trainer.data to persist after
# the Colab runtime is restarted
universe = None

# Changing this to True causes traceback of certain error messages to be hidden
quiet = False

trainer = BasicTrainer(universe, quiet=quiet)

# Path to a Mesh Transformer JAX model, or the model ID of a Hugging Face model
# such as "KoboldAI/fairseq-dense-13B"
trainer.data.ckpt_path = "/content/step_383500"
trainer.get_hf_checkpoint_metadata()

# These two lines below are only required if you're loading from a Mesh
# Transformer JAX model, see the demo notebook for the full list of permitted
# model types
model_type = "GPT-J-6B"
trainer.set_params(model_type)

# Location of the save file (if the file does not exist it will be created), you
# can specify the path to an existing save file created by mtj-softtuner to
# continue from an earlier point in the training
trainer.data.save_file = "/content/my_softprompt.mtjsp"

# Set the initial soft prompt string, this will be ignored if we are continuing
# from an existing save file
initial_softprompt = (
    "Le Jeu du Prochain Train itself is simplicity in motion. The object: "
    "Be the last of your round's six to jump from one side of the tracks to "
    "the other - that is, across the tracks - before the train passes.\n\n"
)
tokenizer = trainer.get_tokenizer()
if trainer.data.newlinemode == "s":  # Handle fairseq-style newlines if required
    initial_softprompt = initial_softprompt.replace("\n", "</s>")
trainer.data.initial_softprompt = tokenizer.encode(
    initial_softprompt, max_length=int(2e9), truncation=True
)

# Do this to generate an NPY file for your dataset if you haven't already done so
dataset_path = "/content/dataset.txt"  # Can be a single file or a folder
output_file = "/content/dataset.npy"
batch_size = 2048
epochs = 1
trainer.tokenize_dataset(dataset_path, output_file, batch_size, epochs)

dataset_file = output_file
trainer.data.dataset_file = dataset_file

trainer.data.gradient_accumulation_steps = 16

# Set training hyperparameters here; see the demo notebook for explanation of
# what these mean
trainer.data.stparams = {
    "lr": 3e-5,
    "max_grad_norm": 10.0,
    "weight_decay": 0.1,
    "warmup": 0.1,
    "end_lr_multiplier": 0.1,
    "save_every": 50,
}

# Now, begin training!
trainer.train()

# Export to KoboldAI/mkultra format
output_file = "/content/my_softprompt.zip"
name = "Untitled"
author = ""
supported = "Generic 6B"
description = "Baby shoes"
trainer.export_to_kobold(output_file, name, author, supported, description)
output_file = "/content/my_softprompt.json"
soft_prompt_name = "Untitled"
soft_prompt_description = "Baby shoes"
trainer.export_to_mkultra(output_file, soft_prompt_name, soft_prompt_description)

```
