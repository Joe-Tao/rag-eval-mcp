# Retrieval eval: CUAD test split

Search is restricted to the question's own contract. Top 10 chunks per query.

| method | hit@1 | hit@5 | recall@5 | recall@10 | mrr@10 | p50 latency (ms) |
|---|---|---|---|---|---|---|
| bm25 | 0.382 | 0.652 | 0.494 | 0.622 | 0.492 | 1 |
| vector | 0.342 | 0.590 | 0.446 | 0.588 | 0.449 | 33 |
| hybrid | 0.416 | 0.664 | 0.508 | 0.642 | 0.521 | 31 |
| hybrid_rerank | 0.461 | 0.727 | 0.573 | 0.694 | 0.576 | 308 |
| random (expected) | | 0.240 | | | | |

## hit@5 by clause category

| category (hit@5) | n | bm25 | vector | hybrid | hybrid_rerank |
|---|---|---|---|---|---|
| Document Name | 102 | 0.30 | 0.27 | 0.29 | 0.52 |
| Parties | 102 | 0.26 | 0.32 | 0.31 | 0.33 |
| Agreement Date | 93 | 0.40 | 0.20 | 0.31 | 0.72 |
| Governing Law | 83 | 0.90 | 0.67 | 0.87 | 0.98 |
| Expiration Date | 78 | 0.85 | 0.85 | 0.90 | 0.88 |
| Anti-Assignment | 72 | 0.90 | 0.90 | 0.92 | 0.92 |
| Effective Date | 70 | 0.73 | 0.43 | 0.54 | 0.80 |
| License Grant | 50 | 0.82 | 0.72 | 0.92 | 0.86 |
| Cap On Liability | 44 | 0.80 | 0.84 | 0.86 | 0.86 |
| Audit Rights | 38 | 0.84 | 0.63 | 0.82 | 0.84 |
| Revenue/Profit Sharing | 35 | 0.34 | 0.40 | 0.43 | 0.31 |
| Exclusivity | 33 | 0.61 | 0.24 | 0.48 | 0.52 |
| Insurance | 32 | 0.62 | 0.91 | 0.91 | 0.94 |
| Minimum Commitment | 32 | 0.66 | 0.44 | 0.53 | 0.59 |
| Termination For Convenience | 29 | 0.93 | 1.00 | 0.93 | 0.97 |
| Post-Termination Services | 29 | 0.69 | 0.62 | 0.76 | 0.69 |
| Renewal Term | 26 | 0.96 | 1.00 | 1.00 | 1.00 |
| Change Of Control | 26 | 0.92 | 0.92 | 0.96 | 0.85 |
| Covenant Not To Sue | 24 | 0.54 | 0.29 | 0.46 | 0.42 |
| Non-Compete | 23 | 0.52 | 0.87 | 0.83 | 0.83 |
| Ip Ownership Assignment | 23 | 0.52 | 0.65 | 0.65 | 0.57 |
| Non-Transferable License | 22 | 0.82 | 0.77 | 0.82 | 0.82 |
| Volume Restriction | 17 | 0.35 | 0.24 | 0.29 | 0.06 |
| Rofr/Rofo/Rofn | 17 | 0.76 | 0.71 | 0.82 | 0.76 |
| Notice Period To Terminate Renewal | 16 | 0.75 | 1.00 | 0.81 | 0.94 |
| Competitive Restriction Exception | 16 | 0.56 | 0.62 | 0.62 | 0.69 |
| Liquidated Damages | 14 | 0.71 | 0.50 | 0.71 | 0.71 |
| Uncapped Liability | 13 | 1.00 | 0.92 | 1.00 | 1.00 |
| Irrevocable Or Perpetual License | 13 | 0.92 | 0.77 | 1.00 | 1.00 |
| Affiliate License-Licensee | 12 | 0.83 | 0.75 | 0.83 | 0.92 |
| Warranty Duration | 10 | 0.70 | 0.80 | 0.80 | 0.80 |
| No-Solicit Of Employees | 10 | 0.60 | 0.80 | 1.00 | 0.80 |
| Joint Ip Ownership | 7 | 0.71 | 0.57 | 0.57 | 0.71 |
| No-Solicit Of Customers | 7 | 0.71 | 0.29 | 0.57 | 0.71 |
| Non-Disparagement | 7 | 0.71 | 0.57 | 0.57 | 0.71 |
| Affiliate License-Licensor | 6 | 0.83 | 0.67 | 1.00 | 0.67 |
| Third Party Beneficiary | 6 | 0.50 | 0.67 | 0.67 | 0.83 |
| Most Favored Nation | 3 | 0.67 | 0.67 | 1.00 | 1.00 |
| Unlimited/All-You-Can-Eat-License | 3 | 1.00 | 0.67 | 0.67 | 0.67 |
| Source Code Escrow | 1 | 1.00 | 1.00 | 1.00 | 1.00 |

## Settings

- date: 2026-11-16
- questions: 1244
- limit: None
- seed: 0
- chunk_size: 1000
- chunk_overlap: 200
- candidates: 30
- embedder: sentence-transformers/all-MiniLM-L6-v2
- reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- machine: x86_64 Darwin, Python 3.11.13
