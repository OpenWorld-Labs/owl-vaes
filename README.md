<div align="center">

# 🦉 OWL VAEs

<p align="center">
  This is our codebase for VAE training.
</p>

---

</div>

## Basic Information

To get setup just run `pip install -r requirements.txt`.

- Set an **environment variable** for the `WANDB_USER_NAME` to sync correctly w/ Wandb.
- Set an envvar for the `DEVICE_TYPE` to use non-cuda device.
  To launch training run: setup a config yaml file, then run `python -m train /path/to/config.yaml` (or `torchrun`).

Before commiting, we recommend running installing and running the `pre-commit` hooks:

```bash
pre-commit install # one-time command
pre-commit run --all-files
```

## How Do I Create A New Model?

- add your new model to `owl_vaes/models/{name}.py`
- add any new building blocks you need under `owl_vaes/nn`
- add your model to `owl_vaes/models/__init__.py` and give it an ID

## How Do I Create A New Trainer

- Review `owl_vaes/trainers/base.py` and `owl_vaes/trainers/rec.py` to see general formatting
- Implement your trainer in `owl_vaes/trainers/{name}.py`
- Add to `owl_vaes/trainers/__init__.py` and give it an ID

## How Do I Set Up A New Config?

- See existing configs under `configs` directory for examples.

## Run with Docker:
```
docker build -t owl_vaes .

docker run --gpus all -it \
  -v $HOME/.gitconfig:/root/.gitconfig:ro \
  -v $HOME/.ssh:/root/.ssh:ro \
  owl_vaes /bin/bash
```