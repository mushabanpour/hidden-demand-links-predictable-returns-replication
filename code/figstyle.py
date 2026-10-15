"""figstyle.py -- one grayscale, serif figure theme for every figure in the paper.

Matches the house style of the companion paper: Times serif text, black / dark-gray /
light-gray series distinguished by line style and marker (solid + filled circle, dashed +
open square, dash-dot + triangle), left and bottom axes only with outward ticks, no grid,
a thin black zero line, no panel titles (panels are described in the LaTeX caption), and
frameless legends.
"""
import matplotlib as mpl

BLACK, DGRAY, LGRAY = "#000000", "#555555", "#a6a6a6"
BAR_DARK, BAR_LIGHT = "#3a3a3a", "#a6a6a6"

# series styles in fixed order
S1 = dict(color=BLACK, ls="-", marker="o", mfc=BLACK, mec=BLACK)
S2 = dict(color=DGRAY, ls="--", marker="s", mfc="white", mec=DGRAY)
S3 = dict(color=LGRAY, ls="-.", marker="^", mfc=LGRAY, mec=LGRAY)
SERIES = [S1, S2, S3]


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": BLACK,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })


def zero_line(ax, y=0.0):
    ax.axhline(y, color=BLACK, lw=0.8, zorder=0)


def line(ax, x, y, k, label=None, markevery=None, marker=True):
    st = dict(SERIES[k])
    if not marker:
        st.pop("marker"); st.pop("mfc"); st.pop("mec")
    return ax.plot(x, y, label=label, markevery=markevery, **st)
