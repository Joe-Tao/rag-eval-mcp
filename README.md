# rag-eval-mcp

Work in progress. Hybrid retrieval (BM25 + dense + RRF + cross-encoder rerank) over CUAD contracts,
with a retrieval eval harness and an MCP server. Full README comes at the end of the build.

## Quick start (so far)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[models,dev]"
# Intel Mac (torch stops at 2.2.2): use Python 3.11 and
# pip install -r requirements-intel-mac.txt && pip install -e ".[dev]"
pytest
cp ~/rag/data/test.json data/cuad_test.json
python scripts/build_eval_set.py
python scripts/demo_search.py "Which state's law governs this agreement?" --doc c000
```
