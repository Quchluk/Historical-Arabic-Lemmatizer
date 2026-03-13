# ArLem: Arabic Lemmatization Tool with Embedding-Based Error Correction

## Abstract
This project presents **ArLem**, a specialized computational tool designed for the precision lemmatization of Arabic historical texts...

## 1. Introduction
The complexity of Arabic morphology, characterized by its root-and-pattern system, often results in significant ambiguity during automated lemmatization. Traditional rule-based analyzers frequently misinterpret context, leading to incorrect root assignments. This software addresses these limitations by implementing a two-stage process:
1.  **Statistical Anomaly Detection**: Utilizing vector embeddings to measure semantic coherence within lemma groups.
2.  **Contextual Rectification**: Deploying a state-of-the-art LLM to re-evaluate and correct high-probability errors.

This tool is specifically engineered to identify and fix "root-sharing" errors—instances where words are correctly associated with a root but incorrectly lemmatized due to divergent semantic meanings.

## 2. Methodology
The lemmatization process is structured as follows:

### 2.1 Initial Morphological Analysis
The system initiates processing using CAMeL Tools to generate a baseline set of lemmas for the input corpus. This step provides the foundational morphological data upon which subsequent verification layers operate.

### 2.2 Embedding-Based Anomaly Detection
A critical innovation of this tool is its use of semantic vector space for error detection. The system aggregates all words assigned to a specific lemma and computes their vector embeddings. It then calculates the pairwise cosine similarity between these embeddings.
*   **Hypothesis**: Words sharing a correct lemma should exhibit high semantic similarity (high cosine score).
*   **Detection**: Tokens that demonstrate statistically significant deviations (low cosine similarity) from the cluster centroid are flagged as potential lemmatization errors. These outliers typically represent homographs or distinct semantic concepts that have been conflated.

### 2.3 LLM-Based Correction (OpenRouter)
Upon identification of an anomaly, the tool delegates the correction task to an external Large Language Model via the OpenRouter interface. The system constructs a prompt containing the ambiguous token and its surrounding context, instructing the model to determine the precise lemma. This allows for context-aware disambiguation that purely statistical methods cannot achieve.

## 3. Installation and Requirements

### 3.1 Prerequisites
*   Python 3.8 or higher
*   A valid OpenRouter API key (for LLM access)

### 3.2 Setup Procedure
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/lemmatizer.git
    cd lemmatizer
    ```

2.  **Environment Initialization**
    Establish a virtual environment to ensure dependency isolation.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Dependency Installation**
    Install the package in editable mode.
    ```bash
    make install
    ```

4.  **Resource Acquisition**
    Download the requisite CAMeL Tools datasets.
    ```bash
    python -m camel_tools.data_downloader light
    ```

## 4. Configuration
The system requires specific environment variables for operation.

1.  Initialize the configuration file:
    ```bash
    cp .env.example .env
    ```

2.  Configure parameters in `.env`:
    *   `OPENROUTER_API_KEY`: Mandatory for the correction module.
    *   `OPENROUTER_MODEL`: Specifies the LLM model (e.g., `deepseek/deepseek-chat`).
    *   `DATA_DIR`: Directory path for input corpora.

## 5. Usage

### 5.1 Interactive Analysis
The tool provides a web-based interface for real-time analysis and visualization of lemmatization results.
```bash
make run
```

### 5.2 Batch Processing
For large-scale corpora, use the batch processing modules located in `src/lemmatizer`.
```bash
python -m lemmatizer.processing.batch_processor --input_dir ./data/raw
```

## 6. Project Structure details

The repository is organized into a modular architecture designed for scalability and maintainability.

### 6.1 Source Code (`src/lemmatizer/`)

The core logic is encapsulated within the `src/lemmatizer` package:

*   **`core/`**: Fundamental algorithms and models.
    *   `llm_lemmatizer.py`: Manages interactions with the OpenRouter API for lemma correction.
    *   `model.py`: Defines data structures and model classes used throughout the system.

*   **`processing/`**: Data ingestion and morphological processing modules.
    *   `camel_fast.py` & `camel_db.py`: Interfaces for the CAMeL Tools morphological analyzer, optimized for speed and database interaction respectively.
    *   `llm_corpus.py` & `llm_resolve.py`: Specialized handlers for processing large corpora through the LLM resolution pipeline.

*   **`pipeline/`**: Orchestration logic for the entire workflow.
    *   `unified.py`: The main pipeline entry point that connects morphological analysis, embedding generation, and error correction.
    *   `parallel.py`: Implements parallel processing capabilities to handle large datasets efficiently.

*   **`web/`**: User Interface components.
    *   `app.py`: The Streamlit/Web application that provides the interactive analysis dashboard.

*   **`utils/`**: Helper utilities.
    *   `embed_texts.py`: Functions for generating and managing vector embeddings.

*   **`validation/`**: Quality assurance modules.
    *   `spread.py`: Analyzes the semantic spread of lemmas to calculate validity metrics.
    *   `lemmatization.py`: Validates the output of the lemmatization process against ground truth or heuristics.

*   **`training/`**: (Experimental) Modules for fine-tuning or training custom models.
    *   `advanced.py`: Advanced training routines.

### 6.2 Supporting Directories

*   **`scripts/`**: Standalone utility scripts for setup (e.g., `check_dependencies.py`) and maintenance.
*   **`data/`**: Storage for raw input texts, intermediate JSON/CSV files, and final outputs. (Note: Large data files are git-ignored).
*   **`logs/`**: Directory where application logs (`metadata_extraction.log`, etc.) are stored for debugging and audit trails.

## 7. License
This software is distributed under the MIT License. Refer to the LICENSE file for legal terms.

