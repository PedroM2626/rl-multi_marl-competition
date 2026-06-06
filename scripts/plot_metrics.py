import json
import os
import sys

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("Erro ao importar dependências:", e)
    print("Tente instalar dependências: pip install pandas matplotlib")
    sys.exit(1)


ROOT = os.path.dirname(os.path.dirname(__file__))
METRICS_PATH = os.path.join(ROOT, "data", "metrics", "summary.json")
OUT_DIR = os.path.join(ROOT, "exports", "metrics")
os.makedirs(OUT_DIR, exist_ok=True)


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    teams = j.get("teams", [])
    return pd.DataFrame(teams)


def save_csv(df, path):
    df.to_csv(path, index=False)


def plot_metrics(df, out_path):
    cols = ["win_rate", "eliminations_per_match", "shot_accuracy"]
    labels = ["Win Rate", "Eliminations/Match", "Shot Accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = df["team_name"]

    # Win rate
    axes[0].bar(x, df["win_rate"], color=["#4c72b0", "#55a868", "#c44e52"])
    axes[0].set_title("Win Rate")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("")

    # Eliminations
    axes[1].bar(x, df["eliminations_per_match"], color=["#4c72b0", "#55a868", "#c44e52"])
    axes[1].set_title("Eliminations / Match")

    # Accuracy
    axes[2].bar(x, df["shot_accuracy"], color=["#4c72b0", "#55a868", "#c44e52"])
    axes[2].set_title("Shot Accuracy")
    axes[2].set_ylim(0, df["shot_accuracy"].max() * 1.2)

    for ax in axes:
        ax.set_xticklabels(x, rotation=20)

    plt.tight_layout()
    fig.savefig(out_path)


def main():
    if not os.path.exists(METRICS_PATH):
        print(f"Arquivo de métricas não encontrado: {METRICS_PATH}")
        sys.exit(2)

    df = load_metrics(METRICS_PATH)

    # normalize / format columns
    if "shot_accuracy" in df.columns:
        # already in fraction (0-1) in summary.json
        pass

    csv_path = os.path.join(OUT_DIR, "metrics_summary.csv")
    png_path = os.path.join(OUT_DIR, "metrics_comparison.png")

    save_csv(df, csv_path)
    plot_metrics(df, png_path)

    print("CSV salvo em:", csv_path)
    print("Gráfico salvo em:", png_path)


if __name__ == "__main__":
    main()
