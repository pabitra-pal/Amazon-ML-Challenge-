# Hybrid business entity resolution submission

Recreate the base environment from environment.yml, then run python src/setup_neural.py.
Use the resulting neural Python to run:

    python src/reproduce_neural.py --cache-root /path/to/normalized_v1 --work-root /path/to/new_work --dataset-root /path/to/dataset

First run notebook 01 with the supplied raw dataset to create the normalized cache.
The command rebuilds test lookup tables, scores every test reference with the shipped
encoder and XGBoost model, writes output/ TSVs and validates them against raw test IDs.
Keep work/cache directories outside this package. A CUDA GPU and C99 cc compiler are
required. CPU retrieval/text features and GPU E5/XGBoost scoring are batched. No external
business lookup occurs. All model/tokenizer weights for inference are included.
For training details see notebook 06 and the root methodology document.
