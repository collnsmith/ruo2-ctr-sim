"""The one-page results summary (summary.pdf) and its three figures, from the files in this folder.

    python analysis/2026-11-05_oh_fits/make_summary.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.modules["xraydb"] = None                 # the built-in anomalous table, as in the fits
import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure

from ctrfit.fit.fit import Fit
from ctrfit.fit.global_fit import film_of
from ctrfit.fit.workflow import near_bragg
from ctrfit.project.project import Project

FILM_COLOUR = {"4.1 nm": "#2a78d6", "7 nm": "#eb6834"}       # validated categorical slots 1, 2
FILM_MARK = {"4.1 nm": "o", "7 nm": "s"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
ROD_COLOUR = {"(0 -1)": "#8c8b85", "(0 2)": "#1baf7a"}
POTS = ["-0.238 V", "-0.1 V", "0 V", "0.15 V", "0.35 V", "0.7 V", "OCV"]


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def fig_xoh(out):
    d = json.load(open(HERE / "oh_rods_fit_summary.json"))["results"]
    fig = Figure(figsize=(7.4, 2.25), dpi=200, layout="constrained")
    axs = fig.subplots(1, 3, sharey=True)
    titles = {"A shared per potential": "A  dz_Obr fitted", "B fixed 0": "B  dz_Obr = 0",
              "C Rao by potential": "C  dz_Obr of Rao 2017"}
    for ax, (var, r) in zip(axs, d.items()):
        for j, film in enumerate(("4.1 nm", "7 nm")):
            xs, ys, lo, hi = [], [], [], []
            for i, k in enumerate(POTS):
                q = r["profiles"].get(f"{film} {k}")
                if q:
                    xs.append(i + (j - 0.5) * 0.22)
                    ys.append(q["best"])
                    lo.append(q["best"] - q["lo"])
                    hi.append(q["hi"] - q["best"])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=FILM_MARK[film], ms=4.5, color=FILM_COLOUR[film], ecolor=FILM_COLOUR[film],
                        elinewidth=1.2, capsize=0, mec="white", mew=0.6, label=film)
        for x0 in (1, 2):                                            # the two potentials where the films differ
            ax.axvspan(x0 - 0.4, x0 + 0.4, color="#f3f2ee", zorder=0)
        ax.set_xticks(range(len(POTS)))
        ax.set_xticklabels([k.replace(" V", "") for k in POTS], fontsize=6.5)
        ax.set_xlabel("potential (V vs RHE)", fontsize=7.5)
        ax.set_title(titles[var] + f"   χ² {r['red_chi2']:.1f}", fontsize=7.5, color=INK, loc="left")
        ax.set_ylim(-0.03, 1.05)
        style(ax)
    axs[0].set_ylabel("x_OH (OH share of CUS O)", fontsize=7.5)
    axs[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.savefig(out)


def fig_sensitivity(out):
    p = Project.load(HERE / "both_films_autofit.ctrproj.json")
    m, P, D = p.model, p.model.params, p.datasets
    lead = {}
    for ds in D:
        lead.setdefault(film_of(ds).split("|")[0], ds.scope)
    L = np.arange(0.3, 3.551, 0.02)
    L = L[np.abs(L - 3.0) > 0.04]
    fig = Figure(figsize=(3.6, 2.3), dpi=200, layout="constrained")
    ax = fig.add_subplot(111)
    ax.axvspan(2.5, 3.4, color="#f3f2ee", zorder=0)
    ax.text(2.95, 46, "measured", ha="center", fontsize=6.5, color=MUTED)
    ax.axvline(3.0, color=MUTED, lw=0.8, ls=":")
    for film, sc in lead.items():
        v = dict(P.view(sc))
        v["theta"] = 1.0
        a = m.F(0, 1, L, dict(v, x_OH=0.3))
        b = m.F(0, 1, L, dict(v, x_OH=0.7))
        y = 100 * np.abs(b / a - 1)
        w = 3                                               # running median over 0.14 in L: the trend, not the
        y = np.array([np.median(y[max(0, i - w):i + w + 1]) for i in range(y.size)])   # spikes at fringe minima
        ax.plot(L, y, color=FILM_COLOUR[film], lw=2.0)
        i = int(np.argmin(np.abs(L - 1.6)))
        ax.text(1.62, y[i] + (3.5 if film == "7 nm" else -4.5), film, fontsize=7, color=INK, va="center")
    ax.set_xlabel("L on (0 1) (r.l.u.); Bragg peak at L = 3", fontsize=7.5)
    ax.set_ylabel("|F| change for x_OH 0.3 → 0.7 (%)", fontsize=7.5)
    ax.set_ylim(0, 50)
    style(ax)
    fig.savefig(out)


def fig_processing(out):
    p = Project.load(HERE / "both_films_autofit.ctrproj.json")
    m, D = p.model, p.datasets
    f = Fit(m, D)
    fig = Figure(figsize=(3.6, 2.3), dpi=200, layout="constrained")
    ax = fig.add_subplot(111)
    for ds, F in zip(D, f.model_F()):
        rod = f"({ds.rod_labels[0]})"
        if film_of(ds).split("|")[0] != "4.1 nm" or ds.meta.get("potential_V") != 0.0 or rod not in ROD_COLOUR:
            continue
        keep = ~near_bragg(ds, 0.05, m)
        x, y = np.log10(F[keep]), np.log(ds.F[keep] / F[keep])
        k, c = np.polyfit(x / np.log10(np.e), y, 1)
        ax.scatter(10 ** x, np.exp(y), s=7, color=ROD_COLOUR[rod], edgecolors="white", linewidths=0.3, zorder=3)
        xx = np.linspace(x.min(), x.max(), 20)
        ax.plot(10 ** xx, np.exp(c + k * xx / np.log10(np.e)), color=ROD_COLOUR[rod], lw=1.5)
        ax.text(10 ** xx[-1] * 1.15, np.exp(c + k * xx[-1] / np.log10(np.e)), f"{rod}\nslope {k:+.2f}", fontsize=6.5,
                color=INK, va="center")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("model |F| (e), 4.1 nm film at 0 V", fontsize=7.5)
    ax.set_ylabel("data / model", fontsize=7.5)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 4)
    style(ax)
    fig.savefig(out)


def pdf(out, figs):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"     # DejaVu: has the Greek letters
    pdfmetrics.registerFont(TTFont("DejaVu", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")
    base = ParagraphStyle("b", fontName="DejaVu", fontSize=7.8, leading=9.9, textColor=colors.HexColor(INK))
    small = ParagraphStyle("s", parent=base, fontSize=7, leading=8.6, textColor=colors.HexColor(MUTED))
    head = ParagraphStyle("h", parent=base, fontName="DejaVu-Bold", fontSize=9, leading=11, spaceBefore=3)
    title = ParagraphStyle("t", parent=base, fontName="DejaVu-Bold", fontSize=12, leading=17, spaceAfter=3)
    bullet = ParagraphStyle("bl", parent=base, leftIndent=9, bulletIndent=0, spaceAfter=1.5)
    B = lambda t: Paragraph(t, bullet, bulletText="•")
    story = [
        Paragraph("RuO<sub>2</sub>/TiO<sub>2</sub>(110), 4.1 and 7 nm films in 0.1 M H<sub>2</sub>SO<sub>4</sub>: "
                  "CTR fits for the CUS OH coverage", title),
        Paragraph("26 rod scans (16 of the 4.1 nm film, 10 of the 7 nm film), 2234 points, kinematic model ctrfit; "
                  "fits of 5 Nov 2026 (files in analysis/2026-11-05_oh_fits/). x_OH = OH share of the oxygen on the CUS "
                  "Ru, theta (CUS coverage) fixed at 1, OH 2.07 Å above its Ru, H<sub>2</sub>O height fitted (2.57 to 2.62 Å).",
                  small),
        Spacer(1, 3),
        Paragraph("Results", head),
        B("<b>Films.</b> 4.1 nm fully strained: eps_perp 1.44 to 1.51 %, interface 3.11 to 3.18 Å. 7 nm largely relaxed "
          "along [001]: eps_perp 0.50 to 0.64 %, and its (2 0) rod shows no film fringes (the film is out of registry along "
          "[001]). Both mean thicknesses ended on their fit bounds, so they compensate something; the fringe periods give "
          "about 12.8 and 21.7 trilayers."),
        B("<b>Only the (0 ±1) rods carry OH information</b> (Δχ² 120 to 415 between x_OH 0.3 and 0.7, against 5 or less for "
          "(±1 0), (−2 0), (0 2), (2 0)), and they were measured at L 2.5 to 3.4, next to the L = 3 Bragg peak, where x_OH "
          "changes |F| by only 7 to 10 % (Fig. 2)."),
        B("<b>The absolute OH coverage is not determined.</b> On the 13 (0 ±1) scans the best x_OH follows the assumed bridging-O "
          "height (Fig. 1: about 0 to 0.45 in A, 0.45 to 0.65 in B, 0.75 to 1 in C) at nearly equal χ² (14.5, 14.5, 14.9)."),
        B("<b>Film comparison at the same potential, in all three assumptions:</b> at −0.1 V and 0 V (shaded) the 4.1 nm film "
          "has a higher x_OH than the 7 nm film by about 0.3 to 0.5, i.e. its CUS oxygen sits about 0.2 Å lower on average; at "
          "0.35 and 0.7 V the films agree within error; at OCV the order reverses. Equally possible: the same species sits "
          "lower on the more strained film. Not yet a firm result: reduced χ² is 14.5, and the film thicknesses are on bounds."),
        B("<b>Data processing.</b> The 4.1 nm (0 2) rods have a compressed range at every potential (slope −0.4 of data/model "
          "against intensity, Fig. 3; the (0 ±1) rods are near 0), and rod levels differ 3 to 7× within one film and potential. "
          "Overall misfit: reduced χ² 46.5 on all rods, 18.1 on the well-behaved ones, 14.5 on the OH rods."),
    ]
    W = 7.5 * inch
    f1 = Image(str(figs[0]), width=W, height=W * 2.25 / 7.4)
    f2 = Image(str(figs[1]), width=W / 2 - 4, height=(W / 2 - 4) * 2.3 / 3.6)
    f3 = Image(str(figs[2]), width=W / 2 - 4, height=(W / 2 - 4) * 2.3 / 3.6)
    cap = lambda t: Paragraph(t, small)
    story += [Spacer(1, 3), f1,
              cap("<b>Fig. 1</b> x_OH with its allowed range (profile, Δχ² ≤ 4 × reduced χ²) per film and potential, for three "
                  "assumptions on the bridging-O height (A: one value per potential shared by the films; reduced χ² in the "
                  "panel titles). The level moves with the assumption; the 4.1 nm − 7 nm difference at "
                  "−0.1 and 0 V does not."),
              Spacer(1, 2)]
    t = Table([[f2, f3], [cap("<b>Fig. 2</b> Where OH changes the (0 1) rod (running median over 0.14 in L): about 35 to 45 % at L 0.7 to 1.1, 7 to 10 % in the "
                              "measured range (grey)."),
                          cap("<b>Fig. 3</b> Data / model against model |F|: the (0 2) rod is compressed (strong points too weak, "
                              "weak ones too strong), the (0 −1) rod is not. Points within ±0.05 of a Bragg peak left out.")]],
              colWidths=[W / 2, W / 2])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    story += [t, Paragraph("Next steps", head),
              B("Measure (0 1) at L 0.5 to 2.0 on both films at the same potentials: 3 to 4 times the OH sensitivity (Fig. 2)."),
              B("Stationary CVs: (0 −1 1.25) changes 28 to 37 % with x_OH (so far measured only on the 3.3 nm film); the ratio of "
                "(0 1 2.75) to (0 1 3.16 / 3.25) cancels the scale and isolates OH (opposite signs, −11 to −14 % against +7 to +10 %)."),
              B("Fix the (0 2) processing (attenuator transmission near L 1.86, detector count rate, background at the weak ends) and "
                "the rod normalisation; then refit with the thicknesses fixed at their fringe values.")]
    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.5 * inch, rightMargin=0.5 * inch, topMargin=0.42 * inch,
                            bottomMargin=0.35 * inch, title="CTR fits: OH coverage summary", author="ctrfit")
    doc.build(story)


def main():
    figs = [HERE / "fig1_xoh_by_potential.png", HERE / "fig2_oh_sensitivity_01.png", HERE / "fig3_processing_02.png"]
    fig_xoh(figs[0])
    fig_sensitivity(figs[1])
    fig_processing(figs[2])
    pdf(HERE / "summary.pdf", figs)
    print("wrote", HERE / "summary.pdf")


if __name__ == "__main__":
    main()
