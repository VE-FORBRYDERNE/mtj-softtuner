import mtj_softtuner
import mtj_softtuner.testing_utils
import os
import requests


@mtj_softtuner.testing_utils.core_fully_initialized
def test_basic_trainer():
    """
    This test just makes sure that the BasicTrainer example can be run on a
    CPU without crashing or raising an error.
    """
    if os.path.isfile("my_softprompt.mtjsp"):
        os.remove("my_softprompt.mtjsp")
    data = requests.get(
        "https://archive.org/download/TheLibraryOfBabel/babel_djvu.txt"
    ).content.decode("utf8")
    with open("dataset.txt", "w") as f:
        f.write(data)

    trainer = mtj_softtuner.BasicTrainer()
    trainer.data.ckpt_path = "KoboldAI/fairseq-dense-125M"
    trainer.get_hf_checkpoint_metadata()
    trainer.data.save_file = "my_softprompt.mtjsp"
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
    dataset_path = "dataset.txt"
    output_file = "dataset.npy"
    batch_size = 128
    epochs = 1
    trainer.tokenize_dataset(dataset_path, output_file, batch_size, epochs)
    dataset_file = output_file
    trainer.data.dataset_file = dataset_file
    trainer.data.gradient_accumulation_steps = 3
    trainer.data.stparams = {
        "lr": 3e-5,
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
