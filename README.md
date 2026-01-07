# Diploma Thesis – Documentation

## Introduction

- **Title**: Exploring language-based representations for outcome prediction in League
of Legends

- **Project overview**: This thesis investigates whether embedding representations of League of Legends
matches, organized into historical sequences, can be used to predict match outcomes.
Exploratory analysis and quantitative experiments are conducted to evaluate this hypothesis.
In parallel, large language model–based prediction experiments are explored.


- **What's in this repo**: This repository contains the code and datasets used in the diploma thesis, 
as well as two exploratory Jupyter notebooks.

- **What's in this README**: The README provides documentation for navigating the repository and
running the experiments from both the user and technical perspectives.

## Quickstart

To begin, complete the following preparatory steps:

- Unpack the datasets provided in the accompanying ZIP attachment of the thesis and place them in the root folder.
- Prepare a Python environment and install the required dependencies listed in
  the project’s requirements file.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

After completing the steps above, there are two ways to explore and reproduce the
work presented in this thesis.

---

### Quickstart A: Interactive exploration via Jupyter notebooks

This option allows readers to explore the experiments interactively using the
provided Jupyter notebooks. The notebooks load the final datasets and reproduce
the key analyses, figures, and diagnostic experiments discussed in the thesis.

This workflow is suitable for readers who want to:

- Inspect intermediate results interactively
- Reproduce figures without rerunning full training pipelines

Jupyter is included as a dependency in the project environment.

- For the embedding chapters (4,5) refer to **predictions/example.ipynb** (limited to exploration - 
for model training refer to the Quickstart B)
- For the LLM chapter (6) refer to **llm_predictions/example.ipynb**

---

### Quickstart B: Re-running experiments via scripts

This option allows readers to reproduce the experiments by running the Python
scripts included in the repository. It covers the baseline models, the
embedding-based sequence models, and the LLM-based prediction pipeline.

This workflow mirrors the experimental setup used to generate the reported
results and is suitable for readers who wish to:

- Rerun training and evaluation procedures
- Modify hyperparameters or configurations
- Inspect model behavior through direct execution

Concrete command-line invocations used to generate the reported results are provided in the thesis appendix
___

Although it would be possible to download the raw data and rerun the full
preprocessing pipeline, this repository provides the **final datasets** directly.
This design choice is motivated by the following considerations:

- Data preparation was performed sequentially rather than as a single end-to-end
  pipeline.
- Several intermediate steps—such as scraping, ASR correction, and embedding
  extraction—involve repeated data copies and are not optimized for
  re-execution.
- Providing the final datasets allows reviewers to focus directly on the
  experimental results rather than on data reconstruction.


---

## Repository structure

The repository is organized into the following top-level folders:

- **`data_scraping/`**  
  Scripts used to download raw match data and commentary transcripts.

- **`data_integration/`**  
  Scripts for cleaning, merging, and aligning multiple data sources into a
  unified dataset.

- **`asr/`**  
  Post-processing and correction of YouTube automatic speech recognition
  (ASR) outputs.

- **`data_analysis/`**  
  Exploratory analysis and sanity checks performed during development.

- **`predictions/`**  
  LSTM-based prediction models, including:
  - random and history-based baselines,
  - numerical-feature models,
  - embedding-based models.

- **`llm_predictions/`**  
  Experiments using large language models (LLMs) for match outcome prediction
  via prompting.

---
## Chapter to repository mapping

- **Chapter 3 — Data, datasets and our modeling approach**  
  - `data_scraping`  
  - `asr`  
  - `data_integration` *(tabular dataset construction)*
  - `predictions` (for the LSTM definition: file `lstm.py` and 
`sequence_dataset.py` for the sequential dataset construction)

- **Chapter 4 — Task definition, lower and upper bounds on performance**  
  - `predictions`

- **Chapter 5 — Do Raw Text Embeddings Carry Predictive Signal?**  
  - `data_integration` *(embeddings, embedding preprocessing)*  
  - `data_analysis`  
  - `predictions`

- **Chapter 6 — Large Language Models for Match Outcome Prediction**  
  - `llm_predictions`


---

## User documentation

User-facing scripts are located primarily in:

- `predictions/`
- `data_analysis/`

Each experiment can be run via Python scripts with command-line arguments.
Example commands are provided in the thesis.
Datasets are assumed to be already present in the expected locations as mentioned in the quickstart.

### `data_analysis/embedding_pca_analysis.py`

This module is used to reproduce the **PCA-based embedding analyses** presented
in the thesis.



The script produces multiple PCA plots, including:
- team-vs-team comparisons using half-markers,
- categorical coloring by metadata (e.g. year),
- continuous coloring by numerical match properties (e.g. kill difference,
  game length).

The embeddings used can be changed by selecting a different input dataset.
The metadata used for coloring can be modified via the `color_cols` parameter of the following functions:

pca_full_markers_pipeline;
pca_half_markers_pipeline

### `llm_predictions`

The outputs used in the thesis are provided in `llm_predictions/`, as re-running the code requires an API key.
The Jupyter notebook `llm_predictions/example.ipynb` contains code for inspecting these outputs.



### `predictions/run_experiment.py`

This script is the core experimental driver used in Chapters 4 and 5 of the
thesis. It runs the specified training procedure, reports the best validation
metric observed in each run, saves the best-performing model, and produces
training curves.

In its basic form, the script can be executed without parameters
(`python run_experiment.py`), in which case a default configuration is used.

The primary configuration parameters are `--dataset`, which specifies the input
dataset, and `--optuna_run`, which toggles between standard training and
hyperparameter optimization mode. When `--optuna_run` is enabled, selected
hyperparameters are overridden by Optuna-suggested values before the final
training runs. Trials may also be terminated early via Optuna’s pruning
mechanism if intermediate performance is poor.

The prediction target is selected via `--target_col`, which supports
`team1_result`, `team1_kills`, `gamelength`, and `kill_total`.
The `--feature_fn` parameter controls which feature sets are used as model
inputs:
- `embedding`: text embeddings only,
- `target`: history of the target variable only,
- `numerical`: numerical match features,
- `all`: numerical features combined with embeddings,
- `garbage`: random inputs (random baseline).

For embedding-based configurations, dimensionality can be reduced using PCA via
the `--pca` parameter.

Training samples are constructed as fixed-length **historical sequences**.
For each target match, the model receives the previous `k` matches for both
teams, matched by exact player roster and ordered chronologically, with only
matches strictly preceding the target included. The `--k` parameter controls
the maximum history length, while `--min_history` specifies the minimum number
of historical matches required for both teams for a sample to be included.

To control training stability and variance, the number of epochs is set via
`--epochs`, and the entire training process can be repeated multiple times using
`--runs`. Reported results correspond to the mean and standard deviation across
these independent runs.

Remaining parameters correspond to standard LSTM and optimization
hyperparameters. Examples of full script invocations with concrete parameter
settings are provided in the thesis appendix.

An example of a run with rich configuration looks like this:
```
python predictions/run_experiment.py --target_col team1_result --feature_fn numerical --pca 600 --hidden_dim 256 --num_layers 3 --dropout 0.297 --lr 0.0005 --weight_decay 0.000004 --batch_size 64 --epochs 100
 --runs 10
```
Further details (hyperparameters, configurations, and full experiment
descriptions) are provided in the thesis appendices.


---

## Technical documentation

Technical details are documented directly in the code and organized by folder:

- `data_scraping/`: data sources, scraping logic
- `data_integration/`: schema alignment and feature construction
- `asr/`: ASR correction rules and heuristics
- `predictions/`: model definitions, loss functions, training loops
- `llm_predictions/`: prompt templates and evaluation logic

### `data_scraping/`

Technical scripts documenting how raw match commentary was collected from
YouTube.

---

#### Data sources

Commentary is sourced from official YouTube playlists for major League of
Legends tournaments (Worlds, MSI, LCK, LPL, LEC).

Playlist URLs are defined in `youtube_channels.py`, mapping tournament / split
identifiers to league-specific output folders.

---

#### Subtitle download

`youtube_transcript_downloader.py` downloads automatic English subtitles using
`yt-dlp`.

- One folder per league under `commentary_data/`
- `.en.srt` subtitle files and `.info.json` metadata
- Videos are not downloaded; existing folders are skipped

---

#### Subtitle preprocessing

`srt_to_text.py` converts `.en.srt` files to plain text by removing timestamps
and concatenating subtitle segments into `.en.plain.txt` files.

The resulting text is used in downstream ASR correction and modeling.

### `data_integration/`

Scripts for cleaning, aligning, and merging multiple data sources
(Oracle’s Elixir match data and YouTube commentary text) into a unified dataset.

This stage documents how the final modeling datasets were constructed.

The outputs of this folder are **final, ready-to-use datasets** consumed by
downstream experiments (`predictions/`, `llm_predictions/`).

These resulting CSV / Parquet files are assumed to already be present in the repository.

#### Oracle Elixir preprocessing

`postprocess_oracle_elixir_data.py` converts raw Oracle’s Elixir match-level data
into a compact, match-centric schema.

- Aggregates per-player and per-team rows into a single row per game
- Extracts:
  - match metadata (date, league, split, patch, duration),
  - team-level statistics (kills, objectives, gold, towers),
  - player and champion lineups per team
- Outputs one processed CSV per league and year

This produces the structured metadata used throughout the thesis.

---

#### Linking commentary to match metadata

`link_youtube_to_oracle.py` aligns YouTube commentary transcripts with Oracle
Elixir matches.

- Parses team names and game order from video filenames
- Uses upload date, teams, and game order to match commentary to Oracle data
- Resolves team abbreviations via a manually curated mapping
- Produces a combined dataset containing:
  - match metadata,
  - full commentary text,
  - video identifiers

This step bridges unstructured commentary text with structured match data.

---

#### Text embedding integration

`embed_texts.py` augments the integrated dataset with vector representations of
commentary text.

- Optionally masks team, player, and champion names
- Supports both HuggingFace and OpenAI embedding models
- Handles long texts via token-based chunking and pooling
- Stores embeddings alongside metadata and masked text

Embeddings generated here are used directly by prediction models.

---

#### Configuration management

`text_and_embedding_config.py` defines immutable configuration objects used to
parameterize preprocessing and embedding.

- Controls masking strategy, ASR variant, model choice, chunking, and pooling
- Generates deterministic, human-readable identifiers for datasets
- Ensures consistent naming and traceability across experiments

 
### `data_analysis/embedding_pca_analysis.py`

PCA-based analysis of text embeddings for structural inspection and comparison.

The script operates on datasets containing fixed-length embedding vectors and
match metadata. Embeddings are projected into a shared low-dimensional space
using `sklearn` PCA, with coordinates attached back to the dataset for reuse
across analyses.

Visualization uses `matplotlib` to support categorical and continuous coloring,
with optional outlier filtering and legend pruning.

This module is used to analyze clustering, variation, and extremes in the
embedding space.

### `asr/`
In the file `test_and_choose_gpt_model.py`, a manually created JSON Lines (`.jsonl`) file containing
commentary transcripts is used as input to OpenAI API calls. Multiple models (e.g., `gpt5`, `gpt5_mini` ...) 
are queried, and each model’s output is appended to an output file, also stored in `.jsonl` format.

Next, `asr_evaluation.py` evaluates all candidate models on a 
development set (`asr_error_correction_deveset_corrected.jsonl`) using metrics from the `jiwer`, `nltk.translate`, 
and `bert_score` Python modules. The evaluation results are aggregated and saved as a dataframe.

Finally, in `run_batch_gpt.py`, the selected model is applied to the full dataset by rerunning the pipeline end-to-end.
This produces and saves a dataframe in which the transcript text column is added or replaced with the corrected output.



### `llm_predictions/`

This folder contains the large language model (LLM)–based pipelines used in the
thesis to extract structured match information from commentary transcripts and
to predict match outcomes via prompting.  
All model outputs used in the thesis are provided directly.

The pipeline consists of three stages: Pass 1 event extraction, sequence
construction, and Pass 2 outcome prediction.

---

#### Pass 1: Commentary → structured events  
**File:** `run_batch_gpt_pass1.py`

This script converts a full League of Legends match commentary transcript into
an ordered list of influential events. Each event is a continuous substring of
the original text and is assigned one label from:

`teamfight`, `objective_kill`, `team_performance`,
`player_performance`, `match_winner`.

Transcripts are processed as a whole (no sliding-window chunking) and inferred
via batch jobs using the OpenAI Responses API. Outputs are validated and stored
as JSON-encoded event lists appended to the dataset.

---

#### Sequence construction for Pass 2  
**File:** `prepare_pass2_seq.py`

This script builds roster-aware historical sequences from Pass 1 outputs.  
For each target match, it retrieves the previous `k` matches for each team with
the same roster, strictly earlier in time, and attaches their event lists and
outcomes.

The resulting JSONL dataset contains:
- target match metadata and ground-truth label,
- ordered histories for both teams (oldest → newest).

This representation mirrors the history construction used in baseline models.

---

#### Pass 2: Sequential histories → match outcome  
**File:** `run_batch_gpt_pass2.py`

This script predicts match outcomes directly from the sequential histories
produced above. One prompt is generated per target match, containing both teams’
historical event sequences and outcomes.

The model is constrained to output a **single binary digit** (`1` if team1 wins,
`0` otherwise). Inference is performed in batches, and predictions are stored
along with raw outputs and target metadata.

The output is also a JSONL file.

---

#### Reproducibility notes

- All LLM-based scripts require external API access.
- Final extracted events and predictions used in the thesis are provided
  directly in the repository and explored in the **example.ipynb** notebook.


### `predictions/run_experiment.py`

This script orchestrates the full training, evaluation, and reporting pipeline
used for all LSTM-based baselines in the thesis. It is designed as a single,
self-contained experiment driver that combines data preparation, model
training, hyperparameter search, and result logging.

The script serves as an executable experiment specification whose behavior is controlled entirely
via command-line arguments.
The individual components are extracted to their own files for easier maintenance and readability (
e.g. `lstm.py` and `sequence_dataset.py` for the LSTM definition and dataset
construction, respectively
).

---

#### External libraries and dependencies

The implementation relies on the following core libraries:

- **PyTorch** (`torch`, `torch.nn`, `torch.optim`)  
  Used for defining the LSTM model, loss functions, optimization, and training
  loops.

- **NumPy**  
  Used for numerical operations, aggregation of metrics, and deterministic
  seeding where applicable.

- **Pandas**  
  Used for loading datasets, manipulating tabular match data, and exporting
  results.

- **scikit-learn**  
  Provides PCA for optional embedding dimensionality reduction and utilities
  for data normalization.

- **Optuna**  
  Used for automated hyperparameter optimization when `--optuna_run` is
  enabled, including trial pruning based on intermediate validation results.

- **Matplotlib**  
  Used to generate and save training and validation curves.

The script assumes that datasets have already been fully preprocessed and are
available locally in CSV or Parquet format.

---

#### High-level execution flow

At a high level, execution proceeds through the following stages:

1. **Argument parsing and configuration**  
   Command-line arguments are parsed to determine the dataset, target variable,
   feature configuration, training hyperparameters, and whether Optuna-based
   optimization is enabled. In Optuna mode, selected arguments are overwritten
   by trial-specific values.

2. **Dataset loading and preprocessing**  
   The dataset is loaded using Pandas. Feature preprocessing is applied
   according to the selected `feature_fn`, including optional PCA projection
   for embedding features.

3. **Sequence construction**  
   Match-level rows are converted into fixed-length sequential samples.
   For each target match, historical matches are retrieved in chronological
   order, grouped by exact team roster, and truncated or filtered according to
   `k` and `min_history`. These sequences form the inputs to the LSTM.

4. **Model initialization**  
   An LSTM-based sequence model is instantiated using the specified architecture
   parameters (hidden size, number of layers, bidirectionality, pooling
   strategy). Output heads differ depending on whether the task is classification
   or regression.

5. **Training and validation loop**  
   The model is trained for a fixed number of epochs, with validation performed
   after each epoch. The best-performing model checkpoint is tracked according
   to the selected validation metric.

6. **Repetition and aggregation**  
   The entire training procedure can be repeated multiple times (`--runs`) to
   account for non-determinism. Metrics are aggregated across runs and reported
   as mean and standard deviation.

7. **Reporting and persistence**  
   Training curves are saved as figures, and the best model checkpoint is stored
   to disk. Final performance summaries are printed to standard output.

---

#### Hyperparameter optimization flow

When Optuna optimization is enabled:

- A study is created with a user-specified objective metric.
- Each trial samples a configuration and runs a shortened training loop.
- Poorly performing trials may be terminated early via pruning.
- After optimization, the best trial’s parameters are used for the final
  training runs reported in the thesis.

This mode reuses the same data loading and training logic as standard runs,
ensuring comparability between optimized and non-optimized experiments.


