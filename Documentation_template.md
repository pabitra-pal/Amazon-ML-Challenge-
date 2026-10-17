# Business Entity Resolution Solution

**Team Name:Jay Jagannath
**Team Members:Pabitra Pal,Jagadish Prasad Pattanaik,Pankaj Maity,Shubhashree Mohanty

## 1. Executive Summary
Lexical candidate retrieval plus an E5 multilingual shared encoder and XGBoost pair classifier.
Selected model: 05_hybrid. Outputs passed full raw-ID validation.

## 2. Methodology
Original multilingual text is preserved. Offline ICU normalization/transliteration creates
additional comparison views. E5 text includes original name/address and romanized views,
with query prefix, per-field character caps and 192-token truncation. No external entity
lookup, geocoding or external business data are used. ICU is a preprocessing dependency;
its use is our interpretation of the rules, not an asserted organizer approval.

## 3. Candidate Generation
Same-country name/address/trigram/joint lexical retrieval uses caps 20/20/10/10 and exact
BM25 scoring. No true matches are injected into evaluation or test candidates. The exported
candidate lists are exactly the pairs scored. Multiple matches and empty lists are supported.

## 4. Matching Model
MIT-licensed multilingual-e5-small (118M class of model) and Apache-licensed XGBoost are
below 8B parameters. E5 uses explicit-positive/hard-negative contrastive training, not
implicit in-batch negatives. XGBoost uses 43 lexical features plus embedding cosine similarity.
Larger-run classifier training samples hard negatives; tuning and audit retain all candidates.
The selected threshold is 0.76, with 695 trees.
The shipped encoder and tokenizer define exact inference; no retraining is needed to reproduce.

## 5. Results and Error Analysis
{
  "model": "05_hybrid",
  "audit_macro_f05": 0.9368734841331878,
  "anchors": 10000,
  "precision": 0.9810646314365762,
  "recall": 0.876816808173838,
  "threshold": 0.76,
  "note": "Selection and threshold use tune only. No France labels; not a leaderboard score."
}

The audit does not measure France or leaderboard quality. Tune controls model selection;
the final audit is used for reporting. Inspect run_summary.json for output counts and hashes.

## 6. Reproduction
Run normalization notebook 01, then src/reproduce_neural.py as documented in code README.
Both required TSVs contain every test reference. Source code, environment definitions and
model weights are included. Raw data, work caches and optimizer checkpoints are excluded.
