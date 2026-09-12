# Dual Research Fellow profiles

The same codebase can now be launched as two independent Streamlit processes.

## Korean Research Fellow

- Profile: `korean`
- Default DB: `data/research_fellow.db`
- Default cache: `.cache/research-fellow`
- Existing behavior/data are preserved.

## Paper Submission Research Fellow

- Profile: `paper_en`
- Default DB: `data_paper_en/research_fellow_paper_en.db`
- Default cache: `.cache/research-fellow-paper-en`
- LLM responses default to English.
- This database is intentionally separate from the Korean research workspace.

## Run both together on macOS

```sh
./run_dual_research_fellow.sh
```

It starts:

- Korean: http://localhost:8501
- English paper profile: http://localhost:8502

On macOS the script also opens both browser tabs automatically.

Stop both:

```sh
./stop_dual_research_fellow.sh
```

## Run only one profile

```sh
./run_research_fellow.sh korean 8501
./run_research_fellow.sh paper_en 8502
```

## Optional custom DB/data locations

Korean:

```sh
RESEARCH_FELLOW_PROFILE=korean \
RESEARCH_FELLOW_DATA_DIR=/path/to/korean-data \
streamlit run app.py --server.port 8501
```

English:

```sh
RESEARCH_FELLOW_PROFILE=paper_en \
RESEARCH_FELLOW_DATA_DIR=/path/to/paper-en-data \
streamlit run app_en.py --server.port 8502
```

For server/workspace sync, the English profile uses `RESEARCH_FELLOW_SERVER_DIR_EN`; the Korean profile continues to use `RESEARCH_FELLOW_SERVER_DIR`.
