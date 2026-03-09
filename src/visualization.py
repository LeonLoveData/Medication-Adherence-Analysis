# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from pathlib import Path

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_adherence_distributions(adherence):
    sns.histplot(adherence["pdc"], bins=30, kde=True)
    plt.title("PDC Distribution")
    plt.savefig(FIG_DIR / "pdc_distribution.png")
    plt.close()

    sns.histplot(adherence["mpr"], bins=30, kde=True)
    plt.title("MPR Distribution")
    plt.savefig(FIG_DIR / "mpr_distribution.png")
    plt.close()

    sns.histplot(adherence["max_gap"], bins=30)
    plt.title("Max Refill Gap Distribution")
    plt.savefig(FIG_DIR / "max_gap_distribution.png")
    plt.close()

def plot_model_curves(results):
    for name, (y_test, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr)
        plt.title(f"ROC – {name}")
        plt.savefig(FIG_DIR / f"roc_{name}.png")
        plt.close()

        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(rec, prec)
        plt.title(f"PR – {name}")
        plt.savefig(FIG_DIR / f"pr_{name}.png")
        plt.close()
