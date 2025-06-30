
#!/usr/bin/env python3
"""Select models under TPR-gap^2 constraint and build Pareto frontier.

Usage example:
    python select_pareto.py --grid_csv run_results_bol_saved_educlow_grid.csv --epsilon 0.001 --strategy accuracy --out_csv selected_models.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def build_frontier(df):
    """Return rows on the Pareto frontier (max Accuracy, max Welfare)."""
    df_sorted = df.sort_values('Accuracy', ascending=False)
    frontier = []
    best_welfare = -np.inf
    for _, row in df_sorted.iterrows():
        if row['Welfare'] > best_welfare:
            frontier.append(row)
            best_welfare = row['Welfare']
    return pd.DataFrame(frontier)

def choose_from_frontier(frontier, strategy):
    if frontier.empty:
        return None
    if strategy == 'accuracy':
        return frontier.sort_values('Accuracy', ascending=False).iloc[0]
    elif strategy == 'welfare':
        return frontier.sort_values('Welfare', ascending=False).iloc[0]
    elif strategy == 'ideal':
        # distance to (max Accuracy, max Welfare) after normalization
        acc_norm = (frontier['Accuracy'] - frontier['Accuracy'].min()) / (frontier['Accuracy'].max() - frontier['Accuracy'].min() + 1e-12)
        w_norm = (frontier['Welfare'] - frontier['Welfare'].min()) / (frontier['Welfare'].max() - frontier['Welfare'].min() + 1e-12)
        dist = np.sqrt((1 - acc_norm)**2 + (1 - w_norm)**2)
        return frontier.loc[dist.idxmin()]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_csv', required=True, help='CSV containing all (seed, lambda) results')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='Max allowed TPR_gap_squared')
    parser.add_argument('--strategy', choices=['accuracy', 'welfare', 'ideal'], default='accuracy',
                        help='Selection rule on the Pareto frontier')
    parser.add_argument('--out_csv', default='results/opt_models.csv', help='Where to write the selected rows')
    args = parser.parse_args()

    df = pd.read_csv(args.grid_csv)
    required = {'Accuracy', 'Welfare', 'TPR_gap_squared', 'seed', 'lambda_norm'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required - set(df.columns)}")

    selected_rows = []
    for seed, group in df.groupby('seed'):
        admissible = group[group['TPR_gap_squared'] <= args.epsilon]
        if admissible.empty:
            print(f"WARNING: seed {seed} has no admissible models.")
            continue
        frontier = build_frontier(admissible)
        chosen = choose_from_frontier(frontier, args.strategy)
        selected_rows.append(chosen)

    result = pd.DataFrame(selected_rows)
    if args.out_csv:
        result.to_csv(args.out_csv, index=False)
        print(f"Saved {len(result)} selected models to {args.out_csv}")
    else:
        print(result)

if __name__ == '__main__':
    main()
