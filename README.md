# Diploma Thesis – Code and Documentation

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

- **`chatgpt_predictions/`**  
  Experiments using large language models (LLMs) for match outcome prediction
  via prompting.

---

## User documentation

User-facing scripts are located primarily in:

- `winner_prediction/`
- `chatgpt_predictions/`

Each experiment can be run via Python scripts with command-line arguments.
Example commands are provided in the thesis and in inline comments.

Datasets are assumed to be already present in the expected locations.

---

## Technical documentation

Technical details are documented directly in the code and organized by folder:

- `data_scraping/`: data sources, scraping logic
- `data_integration/`: schema alignment and feature construction
- `asr/`: ASR correction rules and heuristics
- `winner_prediction/`: model definitions, loss functions, training loops
- `chatgpt_predictions/`: prompt templates and evaluation logic

Further details (hyperparameters, configurations, and full experiment
descriptions) are provided in the thesis appendices.
