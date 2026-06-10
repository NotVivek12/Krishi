"""
generate_fl_graphs.py -- AgriFL
================================
Reads existing fl_5_clients.csv, fl_10_clients.csv, fl_20_clients.csv
and produces publication-quality dissertation figures.

Output
------
    reports/metrics/
    ├── fl_accuracy_vs_round_5_clients.png    accuracy curve -- 5 clients
    ├── fl_accuracy_vs_round_10_clients.png   accuracy curve -- 10 clients
    ├── fl_accuracy_vs_round_20_clients.png   accuracy curve -- 20 clients
    ├── fl_client_comparison.png              all 3 on one chart (dissertation)
    └── fl_convergence_summary.png            bar chart: best vs final accuracy

Usage
-----
    cd d:/Krishi
    .venv/Scripts/python.exe reports/generate_fl_graphs.py
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
METRICS_DIR = os.path.join(REPORTS_DIR, 'metrics')
os.makedirs(METRICS_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    5:  '#2196F3',   # blue
    10: '#FF9800',   # orange
    20: '#4CAF50',   # green
}
MARKER   = {5: 'o', 10: 's', 20: '^'}
plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':    11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi':   150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(n_clients: int) -> pd.DataFrame | None:
    path = os.path.join(REPORTS_DIR, f'fl_{n_clients}_clients.csv')
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found.")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {path}  ({len(df)} rounds)")
    return df


def add_smoothed(ax, rounds, values, color, alpha_band=0.12, window=3):
    """Plot raw values faintly + smoothed rolling average."""
    ax.plot(rounds, values, color=color, alpha=0.25, linewidth=1)
    if len(values) >= window:
        smoothed = pd.Series(values).rolling(window, center=True,
                                              min_periods=1).mean().tolist()
    else:
        smoothed = values
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-client individual accuracy plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_individual(df: pd.DataFrame, n_clients: int, save_dir: str):
    """Single accuracy + loss panel for one experiment."""
    rounds = df['round'].tolist()
    accs   = df['accuracy'].tolist()
    losses = df['loss'].tolist()
    color  = PALETTE[n_clients]
    marker = MARKER[n_clients]

    best_idx = int(df['accuracy'].idxmax())
    best_rnd = rounds[best_idx]
    best_acc = accs[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'AgriFL — Federated Learning ({n_clients} Clients, Non-IID)',
                 fontsize=14, fontweight='bold', y=1.02)

    # -- Accuracy panel -------------------------------------------------------
    ax = axes[0]
    smoothed = add_smoothed(ax, rounds, accs, color)
    ax.plot(rounds, smoothed, color=color, linewidth=2.5, marker=marker,
            markersize=5, label='Smoothed accuracy')
    ax.axvline(best_rnd, color='tomato', linestyle='--', linewidth=1.5,
               alpha=0.8, label=f'Best round ({best_rnd})')
    ax.scatter([best_rnd], [best_acc], color='tomato', zorder=5, s=80)
    ax.annotate(f'  Best: {best_acc:.4f}',
                xy=(best_rnd, best_acc), fontsize=9, color='tomato',
                va='bottom')
    ax.set_title(f'Global Accuracy per Round')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Weighted Avg Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(rounds[::max(1, len(rounds)//10)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Loss panel -----------------------------------------------------------
    ax = axes[1]
    smoothed_l = add_smoothed(ax, rounds, losses, 'tomato')
    ax.plot(rounds, smoothed_l, color='tomato', linewidth=2.5, marker=marker,
            markersize=5, label='Smoothed loss')
    ax.set_title(f'Global Loss per Round')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Weighted Avg Loss')
    ax.set_xticks(rounds[::max(1, len(rounds)//10)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'fl_accuracy_vs_round_{n_clients}_clients.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Client comparison overlay chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(dfs: dict, save_dir: str):
    """Overlay all three experiments on one accuracy chart."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('AgriFL — Federated Learning: Client Comparison (Non-IID)',
                 fontsize=14, fontweight='bold')

    for n_clients, df in dfs.items():
        if df is None:
            continue
        rounds = df['round'].tolist()
        accs   = df['accuracy'].tolist()
        losses = df['loss'].tolist()
        color  = PALETTE[n_clients]
        marker = MARKER[n_clients]

        # Accuracy
        smoothed = pd.Series(accs).rolling(3, center=True, min_periods=1).mean().tolist()
        axes[0].plot(rounds, accs, color=color, alpha=0.2, linewidth=1)
        axes[0].plot(rounds, smoothed, color=color, linewidth=2.5,
                     marker=marker, markersize=5, markevery=max(1, len(rounds)//8),
                     label=f'{n_clients} clients (best={max(accs):.4f})')

        # Loss
        smoothed_l = pd.Series(losses).rolling(3, center=True, min_periods=1).mean().tolist()
        axes[1].plot(rounds, losses, color=color, alpha=0.2, linewidth=1)
        axes[1].plot(rounds, smoothed_l, color=color, linewidth=2.5,
                     marker=marker, markersize=5, markevery=max(1, len(rounds)//8),
                     label=f'{n_clients} clients')

    axes[0].set_title('Global Accuracy — All Client Configurations')
    axes[0].set_xlabel('Communication Round')
    axes[0].set_ylabel('Weighted Avg Accuracy')
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Global Loss — All Client Configurations')
    axes[1].set_xlabel('Communication Round')
    axes[1].set_ylabel('Weighted Avg Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'fl_client_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Convergence summary bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_summary(dfs: dict, save_dir: str):
    """
    Grouped bar chart showing best accuracy and final accuracy
    for each client configuration — ideal for dissertation conclusions.
    """
    configs    = []
    best_accs  = []
    final_accs = []
    best_rounds = []

    for n_clients in sorted(dfs.keys()):
        df = dfs[n_clients]
        if df is None:
            continue
        best_idx = int(df['accuracy'].idxmax())
        configs.append(f'{n_clients} Clients')
        best_accs.append(df['accuracy'].max())
        final_accs.append(df['accuracy'].iloc[-1])
        best_rounds.append(df['round'].iloc[best_idx])

    x       = np.arange(len(configs))
    width   = 0.35
    colors_best  = [PALETTE[int(c.split()[0])] for c in configs]
    colors_final = [c + 'AA' for c in colors_best]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, best_accs,  width, label='Best round accuracy',
                   color=colors_best,  edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width/2, final_accs, width, label='Final round accuracy',
                   color=['#90CAF9', '#FFCC80', '#A5D6A7'],
                   edgecolor='white', linewidth=0.8)

    # Annotate values
    for bar, val, rnd in zip(bars1, best_accs, best_rounds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}\n(R{rnd})', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_title('AgriFL — Best vs. Final Accuracy by Client Count',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('FL Configuration')
    ax.set_ylabel('Weighted Avg Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6,
               label='50% baseline')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'fl_convergence_summary.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Print summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(dfs: dict):
    print(f"\n{'='*70}")
    print(f"  AgriFL -- Federated Learning Experiment Summary")
    print(f"{'='*70}")
    print(f"  {'Config':<15} {'Rounds':<8} {'Best Acc':<12} {'Best Round':<12} {'Final Acc':<12} {'Final Loss'}")
    print(f"  {'-'*65}")
    for n_clients in sorted(dfs.keys()):
        df = dfs[n_clients]
        if df is None:
            continue
        best_idx   = int(df['accuracy'].idxmax())
        best_acc   = df['accuracy'].max()
        best_rnd   = df['round'].iloc[best_idx]
        final_acc  = df['accuracy'].iloc[-1]
        final_loss = df['loss'].iloc[-1]
        n_rounds   = len(df)
        print(f"  {str(n_clients)+' clients':<15} {n_rounds:<8} {best_acc:<12.4f} "
              f"{best_rnd:<12} {final_acc:<12.4f} {final_loss:.4f}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("AgriFL — Generating dissertation-quality FL graphs\n")

    # Load all available CSVs
    dfs = {}
    for n in [5, 10, 20]:
        df = load_csv(n)
        if df is not None:
            dfs[n] = df

    if not dfs:
        print("ERROR: No FL CSV files found in reports/. Run federated_train.py first.")
        return

    print(f"\nGenerating per-client plots...")
    for n_clients, df in dfs.items():
        plot_individual(df, n_clients, METRICS_DIR)

    print(f"\nGenerating comparison overlay...")
    plot_comparison(dfs, METRICS_DIR)

    print(f"\nGenerating convergence summary bar chart...")
    plot_convergence_summary(dfs, METRICS_DIR)

    print_summary_table(dfs)

    print(f"All graphs saved to: {METRICS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
