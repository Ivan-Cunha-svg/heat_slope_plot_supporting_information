import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4, "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600, "pdf.fonttype": 42})
def panel(ax, s): ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
