"""One-page PDF: what the in situ SXRD data support without potential control of the 4.1 nm film.
Numbers come from sync_abs.json, curves_sync.json, timeline_rods.json, ctr2cv.json, deep41.json and
plot_data/03_cv_summary/cv_amplitudes.csv. Writes onepager_without_4p1nm_control.pdf (and its figure, onepager_fig.png)."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

J = lambda f: json.load(open(os.path.join(here, f)))
sync, curves, rods, c2c = J("sync_abs.json"), J("curves_sync.json"), J("timeline_rods.json"), J("ctr2cv.json")
deep = J("deep41.json")
amp = pd.read_csv(os.path.join(here, "plot_data", "03_cv_summary", "cv_amplitudes.csv"), comment="#")
R = sync["response"]
summ = sync["summary"]
pct = lambda x: f"{100 * x:+.1f}".replace("-", "−")
pe = lambda x: f"{100 * x:.1f}"

# ------------------------------------------------------------------ figure
COL = {"3.3": "#2a78d6", "4.1": "#eb6834", "7": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
                     "axes.titlesize": 8.5, "legend.fontsize": 7})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.3), gridspec_kw=dict(width_ratios=[1.0, 1.1], wspace=0.25))
X = {"2.75": 0, "3.16": 1, "3.25": 2}
SC = {"3.3": {"2.75": [120], "3.16": [222, 223], "3.25": [125]},
      "7": {"2.75": [311, 341, 343, 344], "3.16": [314], "3.25": [316]},
      "4.1": {"2.75": [496, 497], "3.16": [498], "3.25": [501]}}
OFF = {"3.3": -0.24, "4.1": 0.0, "7": 0.24}
MK = {"3.3": "s", "4.1": "o", "7": "D"}
for f, d in SC.items():
    for L, ns in d.items():
        for j, n in enumerate(ns):
            v = R[str(n)]["logistic"]
            x = X[L] + OFF[f] + (j - (len(ns) - 1) / 2) * 0.08
            hollow = f == "4.1" or n == 344
            a1.errorbar(x, 100 * v["A"], yerr=100 * v["err"], fmt=MK[f], ms=4.0, color=COL[f], mfc="white" if hollow else COL[f],
                        mec=COL[f], mew=1.1, elinewidth=1.0, capsize=0, zorder=3)
a1.axhline(0, color=MUTED, lw=0.7, zorder=1)
a1.set_xticks(range(3)); a1.set_xticklabels(["(0 ±1 2.75)", "(0 ±1 3.16)", "(0 ±1 3.25)"])
a1.set_xlim(-0.55, 2.55); a1.set_ylim(-42, 36)
a1.set_ylabel("change from 0 to 0.7 V (%)")
a1.set_title("(a) Change from 0 to 0.7 V, one point per CV scan")
a1.grid(axis="y", color=GRID, lw=0.5); a1.set_axisbelow(True)
h = [Line2D([], [], marker="s", ls="none", color=COL["3.3"], ms=4.0, label="3.3 nm"),
     Line2D([], [], marker="D", ls="none", color=COL["7"], ms=4.0, label="7 nm (hollow: 0.6° incidence)"),
     Line2D([], [], marker="o", ls="none", color=COL["4.1"], mfc="white", ms=4.0, label="4.1 nm: control, potential unverified")]
a1.legend(handles=h, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.15, fontsize=6.4, labelspacing=0.25)

mid = np.linspace(0, 0.7, 15); mid = 0.5 * (mid[1:] + mid[:-1])
FS = {"3.3": [120, 222, 223], "7": [311, 314, 341, 343, 344]}
for f, ns in FS.items():
    fr = np.array([curves["scans"][str(n)]["raw"]["fraction"] for n in ns]) * 100
    m, se = fr.mean(axis=0), fr.std(axis=0, ddof=1) / np.sqrt(len(ns))
    a2.fill_between(mid, m - se, m + se, color=COL[f], alpha=0.18, lw=0)
    hw = curves["films"][f]["raw"]["halfway"]
    a2.plot(mid, m, "-o", color=COL[f], ms=3, lw=1.6, label=f"{f} nm ({len(ns)} scans): half-way {hw:.2f} V")
    a2.plot([hw, hw], [-12, 50], color=COL[f], lw=0.9, ls=(0, (2, 2)))
a2.axhline(50, color=GRID, lw=0.8, ls="--", zorder=0); a2.axhline(0, color=MUTED, lw=0.6)
a2.set_xlim(0, 0.7); a2.set_ylim(-15, 125)
a2.set_xlabel("E applied (V vs Ag/AgCl; add 0.238 V for RHE)"); a2.set_ylabel("switch completed (%)")
a2.set_title("(b) Switch completed vs potential, both sweeps averaged")
a2.grid(axis="y", color=GRID, lw=0.5); a2.set_axisbelow(True)
a2.legend(loc="lower right", frameon=False, handletextpad=0.4, fontsize=6.6)
figpath = os.path.join(here, "onepager_fig.png")
fig.savefig(figpath, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)

# ------------------------------------------------------------------ numbers for the text
m275, m316, m325 = summ["2.75"]["mean"], summ["3.16"]["mean"], summ["3.25"]["mean"]
hw33, hw7 = curves["films"]["3.3"]["raw"]["halfway"], curves["films"]["7"]["raw"]["halfway"]
sc33 = [x for x in curves["films"]["3.3"]["raw"]["scans"] if x]; sc7 = sorted(x for x in curves["films"]["7"]["raw"]["scans"] if x)
ctrl = amp[amp.kind == "control"].set_index("scan")
c7 = ctrl.loc[356, "sum_pp"]
c33 = ctrl.loc[[126, 127, 227, 135]]
r_rep = (rods["7 nm repeat 0.7 V"], rods["7 nm repeat -0.1 V"])
r_chg = (rods["7 nm 0.7 Va vs -0.1 Va"], rods["7 nm -0.1 Vb -> 0.7 Vb"])
r41 = [v for k, v in rods.items() if k.startswith("4.1 nm") and "OCV" not in k]
tab = {(r["hkl"], r["E"]): r for r in c2c["table"] if r["film"] == "7 nm"}
def rng(r):
    p = sorted((100 * v for v in r["predicted"].values()), key=abs)       # smaller magnitude first
    return f"{p[0]:+.0f} to {p[-1]:+.0f}%".replace("-", "−")
q275, q316 = tab[("(0 +1 2.75)", 0.15)], tab[("(0 +1 3.16)", 0.15)]
q275b, q316b = tab[("(0 +1 2.75)", 0.35)], tab[("(0 +1 3.16)", 0.35)]
steps = deep["partC"]
b41 = {L: (summ[L]["ratio_41_to"]["3.3"], summ[L]["ratio_41_to"]["7"]) for L in ("2.75", "3.16")}
ub = lambda r: 100 * (r[0] + 2 * r[1]) if r[0] >= 0 else 100 * (r[0] + 2 * r[1])

# ------------------------------------------------------------------ PDF
FD = "/usr/share/fonts/truetype/liberation/"
pdfmetrics.registerFont(TTFont("LS", FD + "LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("LS-B", FD + "LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("LS-I", FD + "LiberationSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("LS-BI", FD + "LiberationSans-BoldItalic.ttf"))
from reportlab.lib.fonts import addMapping
addMapping("LS", 0, 0, "LS"); addMapping("LS", 1, 0, "LS-B"); addMapping("LS", 0, 1, "LS-I"); addMapping("LS", 1, 1, "LS-BI")
NAVY = colors.HexColor("#1d3557"); GREYT = colors.HexColor("#55534d")
BS = 7.5
st = dict(
    title=ParagraphStyle("t", fontName="LS-B", fontSize=12.2, leading=15.6, textColor=NAVY, spaceAfter=3),
    sub=ParagraphStyle("s", fontName="LS", fontSize=7.3, leading=9.8, textColor=GREYT),
    h=ParagraphStyle("h", fontName="LS-B", fontSize=8.2, leading=9.8, textColor=NAVY, spaceBefore=3.5, spaceAfter=1.2),
    body=ParagraphStyle("b", fontName="LS", fontSize=BS, leading=9.1),
    bul=ParagraphStyle("u", fontName="LS", fontSize=BS, leading=9.1, leftIndent=8.5, bulletIndent=0, spaceAfter=1.0, bulletFontName="LS", bulletFontSize=BS),
    box=ParagraphStyle("x", fontName="LS", fontSize=7.8, leading=9.5, leftIndent=9, bulletIndent=0, spaceAfter=1.2, bulletFontName="LS", bulletFontSize=7.8),
    foot=ParagraphStyle("f", fontName="LS", fontSize=6.3, leading=7.6, textColor=GREYT),
)
P = lambda t, s="body": Paragraph(t, st[s])
B = lambda t: Paragraph(t, st["bul"], bulletText="•")
N = lambda i, t: Paragraph(t, st["bul"], bulletText=f"{i}.")

story = [
    P("RuO<sub>2</sub>(110) films in 0.1 M H<sub>2</sub>SO<sub>4</sub>: what the in situ SXRD data support, "
      "with potential control of the 4.1 nm film treated as unverified", "title"),
    P("NSLS-II, October 2026. Epitaxial films on TiO<sub>2</sub>(110), 3.3, 4.1 and 7 nm. Ag/AgCl reference; RHE = Ag/AgCl + 0.238 V "
      "(logbook). Stationary CVs at fixed reflections: 30 s at 0 V, then 0 to 0.7 V at 5 mV/s. CTR rods at fixed potentials.", "sub"),
    Spacer(1, 3),
]
box = [Paragraph("<b>Bottom line.</b> The 3.3 and 7 nm film surfaces switch reversibly between two states near "
                 f"{hw7:.2f}-{hw33:.2f} V vs Ag/AgCl (about {hw7 + 0.238:.2f}-{hw33 + 0.238:.2f} V RHE), at the reflections the CTR model marks "
                 "as surface-sensitive, while OH-blind reflections stay flat.", st["box"], bulletText="•"),
       Paragraph("Both films switch at the same potential within the scan spread, although the 7 nm film is partly relaxed and the thinner "
                 "film is expected to be fully strained.",
                 st["box"], bulletText="•"),
       Paragraph("The 4.1 nm film is shown only as a negative control: its current record was lost, so potential control is not claimed. "
                 "Under the same cell, beam, protocol and analysis it shows no signal locked to the potential.", st["box"], bulletText="•")]
bt = Table([[box]], colWidths=[7.55 * inch])
bt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3f9")), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9fb3cc")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
story += [bt, Spacer(1, 3)]
w = 7.55 * inch
from PIL import Image as PILImage
iw, ih = PILImage.open(figpath).size
story += [Image(figpath, width=w, height=w * ih / iw), Spacer(1, 1)]
story.append(P("(a) Error bars: ±1 s.e., the larger of a noise-surrogate and an autocorrelation-robust error. (b) Per scan, 0% is the first bin "
               "(0-0.05 V) and 100% the plateau (0.55-0.7 V); bands: ±1 s.e. over scans.", "foot"))

def mini(rows):
    t = Table(rows, colWidths=[1.12 * inch, 1.05 * inch, 1.05 * inch], hAlign="LEFT")
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "LS", BS), ("FONT", (0, 0), (-1, 0), "LS-B", BS),
                           ("TOPPADDING", (0, 0), (-1, -1), 0.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#9a998f"))]))
    return t

f = lambda m: f"{pct(m[0])} ± {pe(m[1])}%"
left = [
    P("1. The switch, without a model (3.3 and 7 nm)", "h"),
    P("Each X-ray point gets the potential EC-Lab applied at that moment (SPEC and EC-Lab clocks agree within 1-2 s); "
      "per scan, a slow drift plus the change from 0 to 0.7 V is fitted. Weighted per reflection:"),
    Spacer(1, 1.5),
    mini([["reflection", "3.3 nm", "7 nm"], ["(0 ±1 2.75)", f(m275["3.3"]), f(m275["7"])], ["(0 ±1 3.16)", f(m316["3.3"]), f(m316["7"])],
          ["(0 ±1 3.25)", f(m325["3.3"]), f(m325["7"])]]),
    Spacer(1, 2),
    B("<b>Surface fingerprint:</b> 2.75 falls while 3.16 rises, in both films. Beam, absorption or normalization artifacts would move "
      "all reflections the same way."),
    B("<b>Reversible:</b> anodic and cathodic sweeps overlap at 5 mV/s; the swing is the same at 25 mV/s (scan 222)."),
    B("<b>Reproducible:</b> 7 nm (0 1 2.75) measured four times (Ru spot 30, 42, 42, 42%); 3.3 nm (0 −1 3.16) twice."),
    B(f"<b>Surface-specific:</b> OH-blind control reflections stay at their noise: 7 nm (2 0 1.75) {100 * c7:.1f}% peak-to-peak; "
      f"3.3 nm (−2 0 1.75), (−2 0 1), (−1 0 2.32), (1 0 1.75) {100 * c33.sum_pp.min():.1f}-{100 * c33.sum_pp.max():.1f}% "
      f"(point noise {100 * c33.noise_per_image.min():.1f}-{100 * c33.noise_per_image.max():.1f}%)."),
    P("2. The 7 nm CTR rods follow the potential", "h"),
    B(f"Whole (0 1) rod, rms of ln(ratio) with one free scale: 0.7 vs −0.1 V gives {r_chg[0]:.3f} and {r_chg[1]:.3f} (both directions), "
      f"against {min(r_rep):.3f}-{max(r_rep):.3f} for repeats at the same potential hours apart."),
    B("Surface states fitted to the CTRs, under two bridging-O assumptions (one from Rao et al., 2017), predict the sign and onset of the "
      f"7 nm CV changes with nothing adjusted. At 0.15 V: 2.75 predicted {rng(q275)}, measured {pct(q275['measured'])}%; "
      f"3.16 predicted {rng(q316)}, measured {pct(q316['measured'])}%. At 0.35 V, 3.16 matches ({rng(q316b)} vs "
      f"{pct(q316b['measured'])}%); 2.75 is underpredicted ({rng(q275b)} vs {pct(q275b['measured'])}%). 3.25 and the size at 0.7 V "
      "are not reproduced."),
    P("3. Same switch potential in two strain states", "h"),
    B(f"Half-way potential: {hw33:.2f} V (3.3 nm; scans {min(sc33):.2f}-{max(sc33):.2f}) and {hw7:.2f} V (7 nm; scans "
      f"{sc7[0]:.2f}-{sc7[-2]:.2f}, one noisy scan aside) vs Ag/AgCl. Equal within the scan spread."),
    B("7 nm is partly relaxed (CTR out-of-plane strain about 0.5%); the thinner films are strained (4.1 nm CTR 1.4%; 3.3 nm not measured "
      "here). On Jeong et al.'s films (different samples), the work function differs by about 0.3-0.4 eV over this thickness range."),
    B(f"<b>Statement:</b> in these two films the switch potential does not follow the strain state: they differ by "
      f"{abs(hw33 - hw7):.2f} V, within the scan spread."),
]
right = [
    P("4. The 4.1 nm film: a negative control", "h"),
    B(f"<b>Potential control unverified.</b> Every day-4 EC file shows zero current: charge at potential steps "
      f"{steps['4.1']['C_uF']:.3f} ± {steps['4.1']['err']:.3f} µF/V, against {steps['7']['C_uF']:.1f} µF/V on 7 nm. The cause was not "
      "found and the setup cannot be re-examined."),
    B("Same cell type, beam, scan protocol and analysis: no signal locked to the potential in any X-ray channel (all detector ROIs and "
      "SPEC counters, judged against the beam monitors), and the (0 −1) CTR rods at six potentials agree to the repeat level "
      f"(rms {min(r41):.3f}-{max(r41):.3f})."),
    B("So the 3.3 and 7 nm switch is not produced by the cell, the beam or the method."),
    B(f"If the potential did reach the 4.1 nm film, it did not switch: at most {ub(b41['2.75'][0]):.0f}% and {ub(b41['3.16'][0]):.0f}% of "
      f"the 3.3 nm response and {ub(b41['2.75'][1]):.0f}% and {ub(b41['3.16'][1]):.0f}% of the 7 nm response at L 2.75 and 3.16 "
      "(2σ). Show this only as an observation, not as a film difference."),
    P("5. Stated limits (not claimed)", "h"),
    B("Absolute OH coverage: the OH fraction and the bridging-O height trade off in the fits; the relative result is the same under both "
      "assumptions."),
    B("A thickness or strain trend: one sample per thickness."),
    B("A mechanism, and the 3.3 vs 7 nm sign difference at L 3.25 (CV 125 was measured off the rod maximum)."),
    B("A steady anodic current on days 2-3 (µA to tens of µA) is not produced by the beam at the moment (unchanged at 3.6% "
      "transmission); its source is unknown. iR at the switch potential is under 1 mV."),
    P("6. Suggested talk (8 slides)", "h"),
    N(1, "Question and answer up front."),
    N(2, "Films, cell, potential program, measurement order."),
    N(3, "Why (0 ±1) reflections: the 2.75 / 3.16 fingerprint."),
    N(4, "Raw traces: 7 nm next to the 4.1 nm control."),
    N(5, "Response at the applied potential, with controls (panel a)."),
    N(6, "Switch curves: the same potential in both films (panel b)."),
    N(7, "7 nm CTRs and the CTR-to-CV prediction."),
    N(8, "Summary, limits, and what would make it a paper: a theory estimate of the strain shift, lab CVs, second samples."),
]
cols = Table([[left, right]], colWidths=[3.82 * inch, 3.73 * inch])
cols.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (0, 0), 0), ("LEFTPADDING", (1, 0), (1, 0), 8),
                          ("RIGHTPADDING", (0, 0), (0, 0), 9), ("RIGHTPADDING", (1, 0), (1, 0), 0),
                          ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                          ("LINEAFTER", (0, 0), (0, 0), 0.4, colors.HexColor("#d6d4cc"))]))
story += [Spacer(1, 2), cols, Spacer(1, 3),
          P("Data and scripts: analysis/2026-11-06_stationary_cv in the ruo2-ctr-sim repository (sync_abs.py, curves_sync.py, timeline_rods.py, "
            "ctr2cv.py, deep41.py, deep41b.py; this page: make_onepager.py).", "foot")]
out = os.path.join(here, "onepager_without_4p1nm_control.pdf")
doc = SimpleDocTemplate(out, pagesize=letter, leftMargin=0.47 * inch, rightMargin=0.47 * inch, topMargin=0.4 * inch,
                        bottomMargin=0.35 * inch, title="RuO2(110) films: what the SXRD data support",
                        author="ruo2-ctr-sim analysis", subject="One-page summary without 4.1 nm potential control")
doc.build(story)
from pypdf import PdfReader
print("pages:", len(PdfReader(out).pages), out)
