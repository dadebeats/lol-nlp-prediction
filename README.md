# Diploma Thesis – Documentation

This repository contains the code and datasets used in the diploma thesis.
The README provides basic documentation for navigating the repository and
running the experiments.

Where possible, code is linked to thesis chapters via docstrings and folder
names.

## Data and reproducibility

Although it would be possible to download the raw data and rerun the full
preprocessing pipeline, we provide the **final datasets** directly.

Reasons:
- Data preparation was performed sequentially rather than as a clean
  end-to-end pipeline.
- Intermediate steps (scraping, ASR correction, embeddings) involve repeated
  data copies and are not optimized for re-execution.
- Providing final datasets allows reviewers to focus on the experiments.

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

- **`winner_prediction/`**  
  LSTM-based prediction models, including:
  - random and history-based baselines,
  - numerical-feature models,
  - embedding-based models.

- **`llm_predictions/`**  
  Experiments using large language models (LLMs) for match outcome prediction
  via prompting.

---

## User documentation

User-facing scripts are located primarily in:

- `winner_prediction/`
- `llm_predictions/`

Each experiment can be run via Python scripts with command-line arguments.
Example commands are provided in the thesis and in inline comments.

Datasets are assumed to be already present in the expected locations.

## `data_analysis/pca_visualization.py`

This module is used to reproduce the **PCA-based embedding analyses** presented
in the thesis.

It operates on datasets that already contain precomputed text embeddings and
match metadata. Running the file computes a global PCA projection and generates
publication-ready visualizations that illustrate structure in the embedding
space.

The script produces multiple PCA plots, including:
- team-vs-team comparisons using half-markers,
- categorical coloring by metadata (e.g. year),
- continuous coloring by numerical match properties (e.g. kill difference,
  game length).

All outputs are saved as `.png` figures, shown with pyplot and correspond directly to analyses
discussed in the experimental section of the thesis.

User can change the embeddings used by changing the dataset to be loaded.

## `llm_predictions`

We do not recommend using scripts in this folder as there would be need to load credit to the OpenAI API and provide
API key. We provide the outputs the API returned, anyway.

---

## Technical documentation

Technical details are documented directly in the code and organized by folder:

- `data_scraping/`: data sources, scraping logic
- `data_integration/`: schema alignment and feature construction
- `asr/`: ASR correction rules and heuristics
- `winner_prediction/`: model definitions, loss functions, training loops
- `llm_predictions/`: prompt templates and evaluation logic

Further details (hyperparameters, configurations, and full experiment
descriptions) are provided in the thesis appendices.


## `data_scraping/youtube_commentary_data`

Technical scripts documenting how raw match commentary was collected from
YouTube. This stage is not intended to be re-executed by users.

---

### Data sources

Commentary is sourced from official YouTube playlists for major League of
Legends tournaments (Worlds, MSI, LCK, LPL, LEC).

Playlist URLs are defined in `youtube_channels.py`, mapping tournament / split
identifiers to league-specific output folders.

---

### Subtitle download

`youtube_transcript_downloader.py` downloads automatic English subtitles using
`yt-dlp`.

- One folder per league under `commentary_data/`
- `.en.srt` subtitle files and `.info.json` metadata
- Videos are not downloaded; existing folders are skipped

---

### Subtitle preprocessing

`srt_to_text.py` converts `.en.srt` files to plain text by removing timestamps
and concatenating subtitle segments into `.en.plain.txt` files.

The resulting text is used in downstream ASR correction and modeling.

## `data_integration/`

Scripts for cleaning, aligning, and merging multiple data sources
(Oracle’s Elixir match data and YouTube commentary text) into a unified dataset.

This stage documents how the final modeling datasets were constructed.
It is not intended to be re-executed by users.

The outputs of this folder are **final, ready-to-use datasets** consumed by
downstream experiments (`winner_prediction/`, `llm_predictions/`).

Users are **not expected** to run these scripts. The resulting CSV / Parquet
files are assumed to already be present in the repository.

### Oracle Elixir preprocessing

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

### Linking commentary to match metadata

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

### Text embedding integration

`embed_texts.py` augments the integrated dataset with vector representations of
commentary text.

- Optionally masks team, player, and champion names
- Supports both HuggingFace and OpenAI embedding models
- Handles long texts via token-based chunking and pooling
- Stores embeddings alongside metadata and masked text

Embeddings generated here are used directly by prediction models.

---

### Configuration management

`text_and_embedding_config.py` defines immutable configuration objects used to
parameterize preprocessing and embedding.

- Controls masking strategy, ASR variant, model choice, chunking, and pooling
- Generates deterministic, human-readable identifiers for datasets
- Ensures consistent naming and traceability across experiments

---

### Notes on reproducibility

- Data integration was performed sequentially and iteratively
- Intermediate artifacts are not optimized for re-execution
- Final integrated datasets are provided directly with the repository
- 
## `data_analysis/embedding_pca_analysis.py`

PCA-based analysis of text embeddings for structural inspection and comparison.

The script operates on datasets containing fixed-length embedding vectors and
match metadata. Embeddings are projected into a shared low-dimensional space
using `sklearn` PCA, with coordinates attached back to the dataset for reuse
across analyses.

Visualization uses `matplotlib` to support categorical and continuous coloring,
with optional outlier filtering and legend pruning.

This module is used to analyze clustering, variation, and extremes in the
embedding space.

## `asr/`
In the file `test_and_choose_gpt_model.py)` we intake a manually create json lines (.jsonl) file containing the
transcript and we use it as an input to OpenAI API calls, calling the specified models (gpt5, gpt5_mini...).
For each of the models we add their output to the output file which is in .jsonl format as well.

Next in `asr_evaluation.py` we evaluate all the models on our devset (`asr_error_coorection_deveset_corrected.jsonl`) with metrics from the
jiwer
;nltk.translate
;bert_score python modules. We save the results in a dataframe.

Finally in `run_batch_gpt.py`, after choosing which model to proceed with, we run the whole pipeline again and this time on the whole dataset.
This returns and saves a dataframe with added (or replaced) column "text".


## `llm_predictions/`

This folder contains the large language model (LLM)–based pipelines used in the
thesis to extract structured match information from commentary transcripts and
to predict match outcomes via prompting.  
These scripts are **not intended for re-execution by users**; all model outputs
used in the thesis are provided directly.

The pipeline consists of three stages: Pass 1 event extraction, sequence
construction, and Pass 2 outcome prediction.

---

### Pass 1: Commentary → structured events  
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

### Sequence construction for Pass 2  
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

### Pass 2: Sequential histories → winner prediction  
**File:** `run_batch_gpt_pass2.py`

This script predicts match outcomes directly from the sequential histories
produced above. One prompt is generated per target match, containing both teams’
historical event sequences and outcomes.

The model is constrained to output a **single binary digit** (`1` if team1 wins,
`0` otherwise). Inference is performed in batches, and predictions are stored
along with raw outputs and target metadata.

The output is also a JSONL file.

---

### Reproducibility notes

- All LLM-based scripts require external API access and are not meant to be run
  by users or reviewers.
- Final extracted events and predictions used in the thesis are provided
  directly in the repository.
