# The Dialect Tax

This is the codebase for ["The Dialect Tax: Dialectal Biases Persist throughout the Language Modeling Pipeline"](https://arxiv.org/abs/2608.24952), to be published in the *Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing*.


## Installation

To set up the repo, run:

```
git clone git@github.com:socialnlp/dialecttax.git
cd dialecttax
uv sync --extra dev
```

Remember to run `hf auth login` to authenticate with your Hugging Face account.

We pin dependencies in `uv.lock` by default, but we include `environment.yml` if you prefer to create a `conda` env instead:

```
conda env create -f environment.yml
conda activate dialecttax
uv sync --extra dev --active
```

Both `torch` and `torchvision` come from the CUDA 12.9 index, but if you have a machine that needs a different backend, run:

```
uv pip install torch torchvision --torch-backend=auto
uv pip install flash-attn --no-build-isolation
```

### Docker

If you'd like to run a Docker container:

```
docker build -t dialecttax .
docker run --gpus all -it dialecttax bash
```


## Structure

We structure the codebase as such:

```
.
├── configs/                     # {repo} and {hostname} are resolved by load_config()
│   ├── default.yaml             # project paths; also server.yaml, external.yaml
│   ├── <experiment>/
│   │   ├── config.yaml
│   │   ├── <parameters>/
│   │   │   ├── <configurations>.yaml
│   │   │   └── ...
│   │   └── ...
│   └── .../
├── data/
│   ├── datasets/                # vendored raw inputs
│   └── preprocessed/            # ReDial and perturbations
├── secrets/                     # API keys, committed as empty stubs; paste
│   ├── api_key_openrouter.txt   #   each key on the first line locally
│   └── api_key_gcloud.txt
├── scripts/                     # pipeline lanes, run in numeric order
│   ├── 0_cpu_datasets.sh        # install raw inputs, preprocess ReDial
│   ├── 1_cpu_tokens.sh          # generate_tokens + generate_words
│   ├── 2_gpu_embeddings.sh
│   ├── 3_gpu_logits.sh
│   ├── 4_gpu_gradients.sh
│   ├── 5_gpu_layers.sh
│   ├── 6_gpu_characters.sh
│   ├── 7_gpu_rewards.sh
│   └── <lane>/                  # the sweep each lane delegates to
├── analysis/                    # tables and figures; plots/ output is gitignored
├── src/dialecttax/              # library code
├── tests/
├── Dockerfile
├── pyproject.toml
├── LICENSE
└── README.md
```

To run the experiments in the paper, execute the numbered scripts in `scripts/` from `0_cpu_datasets.sh` to `7_gpu_rewards.sh`. To run the analyses, run the Python files in `analysis/`. We include stubs for API keys, to be placed in `secrets/`. Configurations for the repo are placed in `configs/`.


## Data

We include a regenerated set of the data we used in `data/`. To download and regenerate the dataset yourself, run:

```
bash scripts/0_cpu_datasets.sh --config <config_file_name>
```

Note that there is inherent stochasticity in regenerating dataset for the MultiVALUE corpus, so we make it opt-in (`--phase multivalue`).
