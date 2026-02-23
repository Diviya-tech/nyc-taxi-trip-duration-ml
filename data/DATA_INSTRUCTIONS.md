# Data instructions

To keep this repository lightweight (and avoid licensing/size issues), the dataset is not committed to GitHub.

## Option A (Course bundle)
Place the provided file here:

- `data/nyc_taxi_data.npy`

It should contain 4 keys:
- `X_train`, `X_test` (pandas DataFrames)
- `y_train`, `y_test` (pandas Series / 1D arrays)

## Option B (Kaggle)
Download the Kaggle “NYC Taxi Trip Duration” dataset and preprocess it to match the expected format.
You can also adapt `src/preprocessing.py` to your raw CSV schema.

## Verify
From the repo root:

```bash
python -c "import numpy as np; d=np.load('data/nyc_taxi_data.npy', allow_pickle=True).item(); print(d.keys()); print(d['X_train'].shape, d['X_test'].shape)"
```
