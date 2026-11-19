import matplotlib as mpl

BLUE, ORANGE, GREEN, RED = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
PURPLE, BROWN, PINK, GRAY, OLIVE, CYAN = "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"

TERRA  = RED
PEACH  = PINK
TEAL   = BLUE
ROSE   = GREEN
SLATE  = PURPLE
OLIVE_ = OLIVE
SAND   = "#a0a0a0"
INK    = "black"
MUTED  = "#555555"
GRID   = "#dddddd"
BOX_FILL, BOX_EDGE = "#e8e8e8", "none"
ORACLE = "black"
MODEL_A = "black"

FAMILY = {
    "NPO + retain": BLUE, "SimNPO": GREEN, "RMU": PURPLE, "Gradient Difference": OLIVE, "Gradient Ascent": "#aec7e8",
    "NPO w/o retain": "#c7c7c7", "Fine-tune on retain": GRAY, "IdkNLL": ORANGE, "IdkDPO": BROWN, "Task vector": GRAY,
    "Calibrated NPO": OLIVE, "Floor": "#a0a0a0", "Match": GRAY, "Match + offset": "#404040",
    "Anchor": RED, "Anchor (shifted)": PINK, "Anchor (proxy)": RED, "Anchor-shift (proxy)": PINK,
    "CIU": CYAN, "KL to base": "#c7c7c7", "KL to proxy": GRAY,
}
MARKER = {
    "NPO + retain": "o", "SimNPO": "^", "RMU": "v", "Gradient Difference": "D", "Gradient Ascent": "<",
    "NPO w/o retain": ">", "Fine-tune on retain": "x", "IdkNLL": "P", "IdkDPO": "X", "Task vector": "h",
    "Calibrated NPO": "D", "Floor": "p", "Match": "H", "Match + offset": "8",
    "Anchor": "s", "Anchor (shifted)": "d", "Anchor (proxy)": "s", "Anchor-shift (proxy)": "d",
    "CIU": ">", "KL to base": "<", "KL to proxy": "d",
}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "axes.titleweight": "normal", "axes.titlelocation": "center",
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.color": "black", "ytick.color": "black",
    "axes.labelcolor": "black", "text.color": "black", "axes.grid": False,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#bbbbbb", "legend.fancybox": False,
    "lines.linewidth": 1.4, "lines.markersize": 5, "scatter.edgecolors": "none",
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

def panel_label(ax, text, y=None):
    ax.set_title(text)

def retrain_box(ax, x0=-0.2, x1=0.2, y0=0.05, y1=1.0, label=False):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=BOX_FILL, edgecolor="none", zorder=0))
