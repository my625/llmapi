## The repository contains the implementations associated with the paper “Auditing Black-Box LLM APIs via Token Probabilities and Data Provenance: A Case Study in Source Code Generation”.


## What’s included

- **Code**: `icws_github.py` (end-to-end pipeline)
- **Datasets (zipped)**:
  - `human_github_pre2021_python.zip` (**human-written** Python code; GitHub, pre-2021)
  - `generated_python_codes.zip` (**LLM-generated** Python code; generated samples)
- **Supplementary materials**:
  - `sample_metadata.json`: **sample metadata** (as referenced in the article)
  - `data_provenance_graph.png`: **data provenance graph** PNG for the **GPT‑4** example referenced in the article


## Requirements

- Python 3.10+ recommended
- Internet access (for OpenRouter calls)

Install dependencies:

```bash
pip install -U torch transformers sentence-transformers scikit-learn networkx pyvis numpy requests
```

## OpenRouter API key setup (required for Min‑K / judges)

Set the environment variable:

### Windows PowerShell

```powershell
setx OPENROUTER_API_KEY "PUT YOUR API KEY"
```

Then restart your terminal / IDE so the variable is visible.

Notes:
- `icws_github.py` ships with `OPENROUTER_API_KEY = "PUT YOUR API KEY"` so no secrets are committed.
- If no key is set, OpenRouter-based steps will fail (or be skipped depending on which entrypoints you call).

## Dataset layout (102 files)

```text
  human_github_pre2021_python/
    1.py
    ...
  gpt_generated_python_codes/
    gpt_4o/
       1.py
       ...

    gpt_4o_mini/
       1.py
       ...

    gpt_5_pro/
        1.py
        ...

    gpt_35_turbo_instruct/
        1.py
        ...
```

Where:
- `human_github_pre2021_python/` contains human-written code files
- `gpt_generated_python_codes/` contains GPT-generated code files (any GPT tier)



## Running

Because `icws_github.py` is an all-in-one research script, it may contain multiple entrypoints (functions / demo blocks).
The simplest way to start is:

```bash
python icws_github.py
```

If you want me to tailor the README to the **exact command/function** you use for the ICWS evaluation (e.g., “run Min‑K on the 102 files and output a CSV”), tell me the function name or the section you run, and I’ll update the “Running” section accordingly.
