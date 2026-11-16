# data/

Not committed. Put the CUAD v1 test split here:

    cp ~/rag/data/test.json data/cuad_test.json

CUAD (Hendrycks et al., 2021) is licensed CC BY 4.0: https://www.atticusprojectai.org/cuad

`scripts/build_eval_set.py` writes `data/eval/cuad_test_eval.jsonl` from it.
