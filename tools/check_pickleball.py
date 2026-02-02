import os
import pandas as pd
import numpy as np
from collections import Counter

SHOT_FOLDER = 'pickleball_shots'

files = [f for f in sorted(os.listdir(SHOT_FOLDER)) if f.lower().endswith('.csv')]
if not files:
    print("No CSV files found in", SHOT_FOLDER)
    raise SystemExit(1)

col_set = None
timesteps = []
labels = []
bad_files = []

for f in files:
    path = os.path.join(SHOT_FOLDER, f)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        bad_files.append(f + ' (read error: ' + str(e) + ')')
        continue
    cols = tuple(df.columns)
    if col_set is None:
        col_set = cols
    elif cols != col_set:
        bad_files.append(f + ' (header mismatch)')
    timesteps.append(len(df))
    if 'shot' in df.columns:
        labels.append(df['shot'].iloc[0])
    else:
        bad_files.append(f + ' (missing shot column)')

print('Files checked:', len(files))
print('Example header (first 6):', col_set[:6], '..., last:', col_set[-1])
print('Header column count:', len(col_set))
print('Feature columns (excluding shot):', len(col_set)-1)
print('Timesteps per file — min/mean/max:', (min(timesteps), np.mean(timesteps), max(timesteps)))
print('Unique labels and counts:')
print(Counter(labels))
if bad_files:
    print('\nFiles with issues (up to 20):')
    for b in bad_files[:20]:
        print('-', b)
else:
    print('\nNo header or shot-column issues detected.')

# Print a small sample from the first file
first = files[0]
print('\nSample rows (first 3) from', first)
print(pd.read_csv(os.path.join(SHOT_FOLDER, first)).head(3))
