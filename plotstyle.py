import matplotlib as mpl

# Categorical slots 1-3 of a palette validated for colour-vision deficiency
# on all pairs (worst CVD delta-E 9.2). Aqua sits below 3:1 contrast on white,
# so every figure that uses it also labels the series directly or in a legend.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#e4e3df"

NODE_COLORS = {
    "TH_NP15_GEN-APND": BLUE,
    "TH_SP15_GEN-APND": ORANGE,
    "TH_ZP26_GEN-APND": AQUA,
}
NODE_LABELS = {
    "TH_NP15_GEN-APND": "NP15",
    "TH_SP15_GEN-APND": "SP15",
    "TH_ZP26_GEN-APND": "ZP26",
}
ACCENT = INK2


def apply():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.7,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": True,
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10.5,
        "figure.titleweight": "bold",
    })


def headline(fig, title, sub=None, y=1.0):
    """Figure-level title stating the finding, with an optional grey subtitle."""
    dy = 0.21 / fig.get_figheight()
    fig.suptitle(title, x=0.01, y=y + (dy if sub else 0.0), ha="left",
                 va="bottom")
    if sub:
        fig.text(0.01, y + 0.01 / fig.get_figheight(), sub, ha="left",
                 va="bottom", fontsize=8, color=INK2)
