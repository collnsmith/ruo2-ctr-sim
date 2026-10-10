"""Matplotlib drawing for the CTR GUI. Every function draws into an existing Figure."""
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

ELEMENT_COLOR = {"Ru": "tab:blue", "O": "tab:red", "H": "0.75", "Ti": "0.4"}
ELEMENT_SIZE = {"Ru": 220, "O": 120, "H": 35, "Ti": 200}


def is_wide(fig, ratio=1.3):
    """Layout class used to switch between side-by-side and stacked panels."""
    w, h = fig.get_size_inches()
    return w / max(h, 1e-6) >= ratio


def message(fig, text):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, color="0.35", wrap=True,
            transform=ax.transAxes)


def _mark_bragg(ax, L, sub, film, excl=None):
    lo, hi = L.min(), L.max()
    for arr, col in ((sub, "0.45"), (film, "tab:red")):
        for Lb in arr[(arr >= lo) & (arr <= hi)]:
            ax.axvline(Lb, color=col, ls=":", lw=0.9)
            if excl:
                ax.axvspan(Lb - excl, Lb + excl, color="0.93", zorder=0)


def _bragg_handles():
    return [Line2D([], [], color="0.45", ls=":", label="TiO2 Bragg"),
            Line2D([], [], color="tab:red", ls=":", label="RuO2 Bragg")]


def draw_structure(fig, m, comp, n_top=3, repeats=2):
    fig.clear()
    at = m.build_film(comp)
    _, z, _, _, _, _ = m.film_layers(m.n_film)
    ztop = z[m.n_film - 1]
    sel = at["z"] > ztop - n_top * m.D_FILM - 0.5
    gs = fig.add_gridspec(1, 2, width_ratios=[m.A2, m.A1])
    for i, (key, A, lab, view) in enumerate((("f2", m.A2, "[1-10] (Å)", "[001]"),
                                             ("f1", m.A1, "[001] (Å)", "[1-10]"))):
        ax = fig.add_subplot(gs[0, i])
        for el in ("Ru", "O", "H"):
            msk = sel & (at["el"] == el)
            for n in range(repeats):
                ax.scatter((at[key][msk] % 1 + n) * A, at["z"][msk] - ztop,
                           s=ELEMENT_SIZE[el] * at["occ"][msk], c=ELEMENT_COLOR[el], alpha=0.8,
                           edgecolors="k", linewidths=0.3)
        ax.set_xlabel(lab)
        ax.set_title(f"looking along {view}", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel("height above nominal top Ru plane (Å)")
        else:
            ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ELEMENT_COLOR[e], label=e)
                               for e in ("Ru", "O", "H")], fontsize=8, loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(f"Top {n_top} trilayers; marker area follows occupancy. Surface: "
                 + (", ".join(f"{k} {v:g}" for k, v in comp.items()) or "empty CUS"), fontsize=9)


def draw_rod_parts(fig, d, comp_text):
    fig.clear()
    if d is None:
        message(fig, "This rod is outside the reachable q range.\nRaise the energy or the max 2θ.")
        return
    H, K, L = d["H"], d["K"], d["L"]
    axs = fig.subplots(1, 3, sharey=True) if is_wide(fig) else fig.subplots(3, 1, sharex=True)
    film_lab = "RuO2 film + adsorbates" + (" + water" if d["water"] else "")
    for ax, (lab, I, c) in zip(axs, (("TiO2 substrate", d["sub"], "0.3"), (film_lab, d["film"], "tab:blue"),
                                     ("total", d["total"], "k"))):
        ax.semilogy(L, I, color=c, lw=1.1)
        _mark_bragg(ax, L, d["bragg_sub"], d["bragg_film"])
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.25)
    if is_wide(fig):
        for ax in axs:
            ax.set_xlabel("L (r.l.u.)")
        axs[0].set_ylabel("|F|² (e²)")
    else:
        axs[-1].set_xlabel("L (r.l.u.)")
        for ax in axs:
            ax.set_ylabel("|F|² (e²)")
    axs[2].legend(handles=_bragg_handles(), fontsize=7, loc="upper right")
    fig.suptitle(f"({H} {K} L), surface: {comp_text}", fontsize=10)


def draw_compare(fig, d, rel_min, excl):
    fig.clear()
    if d is None:
        message(fig, "This rod is outside the reachable q range.\nRaise the energy or the max 2θ.")
        return
    L, labels = d["L"], list(d["I"])
    ref = labels[0]
    cols = cm.coolwarm(np.linspace(0, 1, len(labels)))
    a, b = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    for c, lab in zip(cols, labels):
        I = d["I"][lab]
        a.semilogy(L, I, color=c, lw=1.2, label=lab)
        b.plot(L, 2 * (I - d["I"][ref]) / (I + d["I"][ref]), color=c, lw=1.2)
    _mark_bragg(a, L, d["bragg_sub"], d["bragg_film"])
    _mark_bragg(b, L, d["bragg_sub"], d["bragg_film"], excl=excl)
    b.axhline(0, color="k", lw=0.6)
    for y in (-rel_min, rel_min):
        b.axhline(y, color="0.5", lw=0.6, ls="--")
    a.set_ylabel("|F|² (e²)")
    a.set_title(f"({d['H']} {d['K']} L)", fontsize=10)
    a.legend(fontsize=8, ncol=min(len(labels), 5))
    b.set_ylabel(f"relative to\n{ref}", fontsize=8)
    b.set_xlabel("L (r.l.u.)")
    for ax in (a, b):
        ax.grid(alpha=0.25)


def draw_hk_map(fig, m, res, counts, metric, physical_aspect, labels, n_labels=6, hint=""):
    fig.clear()
    hq, kq = m.q_max * m.A1 / (2 * np.pi), m.q_max * m.A2 / (2 * np.pi)
    hlim = max(max(H for H, _ in res), int(hq))
    klim = max(max(K for _, K in res), int(kq))
    key = "snr_max" if metric == "snr" else "rel_max"
    vals = [r[key] for r in res.values() if r and r[key] > 0]
    norm = LogNorm(max(min(vals), 1e-3 * max(vals)), max(vals)) if vals else Normalize(0, 1)
    nb_vals = [v for v in counts.values() if v is not None]
    bnorm = Normalize(0, max(nb_vals + [1]))

    pts = {k: [] for k in ("rod", "film", "relaxed", "out", "sens", "gray", "unscanned")}
    for H in range(-hlim, hlim + 1):
        for K in range(-klim, klim + 1):
            n = counts.get((abs(H), abs(K)))
            if n is None:
                pts["out"].append((H, K))
                continue
            pts["rod"].append((H, K, n))
            pts["film"].append((H * m.A1 / m.a1_film, K * m.A2 / m.a2_film))
            if m.relaxed and H != 0:
                pts["relaxed"].append((H * m.A1 / m.a1_top, K * m.A2 / m.a2_film))
            r = res.get((abs(H), abs(K)))
            if r is None:
                pts["unscanned"].append((H, K))
            elif r[key] > 0:
                pts["sens"].append((H, K, r[key]))
            else:
                pts["gray"].append((H, K))

    def xy(name):
        arr = np.array(pts[name], float)
        return arr.reshape(-1, arr.shape[1] if arr.size else 2)

    wide = is_wide(fig, 1.25)
    a, b = fig.subplots(1, 2) if wide else fig.subplots(2, 1)
    for ax in (a, b):
        ax._ctr_map = True
        o = xy("out")
        if len(o):
            ax.scatter(o[:, 0], o[:, 1], marker="x", c="0.75", s=10, zorder=1)
    r_ = xy("rod")
    a.scatter(r_[:, 0], r_[:, 1], c=r_[:, 2], cmap="viridis", norm=bnorm, s=34, edgecolors="k", lw=0.3,
              zorder=3)
    f_ = xy("film")
    a.scatter(f_[:, 0], f_[:, 1], s=75, facecolors="none", edgecolors="tab:red", lw=0.7, zorder=4)
    if pts["relaxed"]:
        x_ = xy("relaxed")
        a.scatter(x_[:, 0], x_[:, 1], s=45, marker="D", facecolors="none", edgecolors="tab:orange",
                  lw=0.8, zorder=4)
    if pts["gray"]:
        g = xy("gray")
        b.scatter(g[:, 0], g[:, 1], s=50, c="0.85", edgecolors="k", lw=0.4, zorder=3)
    if pts["unscanned"]:
        u = xy("unscanned")
        b.scatter(u[:, 0], u[:, 1], s=14, facecolors="none", edgecolors="0.6", lw=0.6, zorder=2)
    if pts["sens"]:
        sv = xy("sens")
        b.scatter(sv[:, 0], sv[:, 1], c=sv[:, 2], cmap="magma", norm=norm, s=50, edgecolors="k", lw=0.4,
                  zorder=3)
    # label the best few rods (first quadrant) with their best L
    best = sorted(((r[key], HK) for HK, r in res.items() if r and r[key] > 0), reverse=True)[:n_labels]
    best = best if wide else []   # too crowded when stacked; hover shows the values
    for _, (H, K) in best:
        Lb = res[(H, K)]["L_snr" if key == "snr_max" else "L_rel"]
        b.annotate(f"L={Lb:.2f}", (H, K), xytext=(4, 3), textcoords="offset points", fontsize=7)
    for ax in (a, b):
        ax.add_patch(Ellipse((0, 0), 2 * hq, 2 * kq, fill=False, ls="--", color="tab:gray"))
        ax.scatter(0, 0, marker="*", s=160, c="gold", edgecolors="k", zorder=5)
        ax.set_xlim(-hlim - 0.7, hlim + 0.7)
        ax.set_ylim(-klim - 0.7, klim + 0.7)
        ax.set_xlabel("H (along [001])")
        ax.set_ylabel("K (along [1-10])")
        ax.grid(alpha=0.25)
        if physical_aspect:
            ax.set_aspect(m.A1 / m.A2, adjustable="box")
    handles = [Line2D([], [], marker="o", ls="", color="tab:green", mec="k", label="TiO2 rod"),
               Line2D([], [], marker="o", ls="", mfc="none", mec="tab:red", label="strained RuO2"),
               Line2D([], [], marker="x", ls="", color="0.7", label="beyond q max")]
    if m.relaxed:
        handles.insert(2, Line2D([], [], marker="D", ls="", mfc="none", mec="tab:orange",
                                 label="relaxed RuO2 part"))
    if wide:  # in the stacked layout the hover readout replaces the legends
        a.legend(handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                 frameon=False)
        b.legend(handles=[Line2D([], [], marker="o", ls="", color="0.85", mec="k", label="below min change"),
                          Line2D([], [], marker="o", ls="", mfc="none", mec="0.6", label="not scanned")],
                 fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.colorbar(cm.ScalarMappable(norm=bnorm, cmap="viridis"), ax=a, shrink=0.6, pad=0.02,
                 label="TiO2 Bragg peaks in L range")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="magma"), ax=b, shrink=0.6, pad=0.02,
                 label="max |dI|/sqrt(I), normalized" if key == "snr_max" else "max |dI|/I")
    a.set_title(f"Rods at {m.p['energy_kev']:g} keV, max 2θ {m.p['two_theta_max']:g}°", fontsize=9)
    b.set_title(f"{labels[0]} vs {labels[1]}" + (f" ({hint})" if hint else ""), fontsize=9)


def draw_ab(fig, r, H, K, labels, mark_L=None, excl=0.15):
    fig.clear()
    if r is None:
        message(fig, "Select a row in the ranking to see that rod.")
        return
    L = r["L"]
    a, b = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    a.semilogy(L, r["IA"], color="tab:blue", label=labels[0])
    a.semilogy(L, r["IB"], color="tab:orange", label=labels[1])
    _mark_bragg(a, L, r["bragg_sub"], r["bragg_film"])
    b.plot(L, r["rel"], color="k", lw=1, label="|dI|/I")
    b.plot(L, r["snr"], color="tab:purple", lw=1, label="SNR (normalized)")
    _mark_bragg(b, L, r["bragg_sub"], r["bragg_film"], excl=excl)
    if mark_L is not None:
        for ax in (a, b):
            ax.axvline(mark_L, color="tab:green", lw=1.2, alpha=0.8)
    a.set_title(f"({H} {K} L)" + (f", selected L = {mark_L:.2f}" if mark_L is not None else ""), fontsize=10)
    a.set_ylabel("|F|² (e²)")
    a.legend(fontsize=8)
    b.set_ylabel("sensitivity")
    b.set_xlabel("L (r.l.u.)")
    b.legend(fontsize=8)
    for ax in (a, b):
        ax.grid(alpha=0.25)


def draw_param_study(fig, d, hk, label, excl=0.15, mark_L=None):
    """One parameter-study row on one rod: |F|^2 for every value (coloured by value, colour bar) and
    the change across the values, (max - min) / mean, in %."""
    fig.clear()
    if d is None or hk not in d.get("I", {}):
        message(fig, "Select a row in the parameter ranking to see its rods.")
        return
    L, I = d["L"][hk], d["I"][hk]
    sub, film = d["bragg"][hk]
    a, b = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    norm = Normalize(min(d["values"]), max(d["values"]))
    cmap = cm.viridis
    for v, y in zip(d["values"], I):
        a.semilogy(L, y, color=cmap(norm(v)), lw=1.1)
    _mark_bragg(a, L, sub, film)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=a, pad=0.01).set_label(label, fontsize=8)
    b.plot(L, 100 * d["rel"][hk], color="k", lw=1)
    _mark_bragg(b, L, sub, film, excl=excl)
    fig.colorbar(sm, ax=b, pad=0.01).ax.set_visible(False)        # keeps the two panels aligned
    if mark_L is not None:
        for ax in (a, b):
            ax.axvline(mark_L, color="tab:green", lw=1.2, alpha=0.8)
    H, K = hk
    a.set_title(f"({H} {K} L): {label} from {d['lo']:g} to {d['hi']:g} in {d['steps']} steps"
                + (f", largest change at L = {mark_L:.2f}" if mark_L is not None else ""), fontsize=10)
    a.set_ylabel("|F|² (e²)")
    a.legend(handles=_bragg_handles(), fontsize=7, loc="upper right")
    b.set_ylabel("change (%)")
    b.set_xlabel("L (r.l.u.)")
    for ax in (a, b):
        ax.grid(alpha=0.25)


def draw_study(fig, out, points, thick, relax, coherent_nm):
    fig.clear()
    wide = is_wide(fig, 1.6)
    axs = np.atleast_1d(fig.subplots(1, len(relax), sharey=True) if wide
                        else fig.subplots(len(relax), 1, sharex=True))
    for ax, R in zip(axs, relax):
        for lab, (H, K, _) in points.items():
            ax.plot(thick, [100 * out[(lab, R, t)] for t in thick], marker="o",
                    ls="-" if H == 0 else "--", label=lab)
        ax.axvline(coherent_nm, color="0.5", lw=0.8, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"relaxation {R:g}", fontsize=10)
        if not wide and ax is axs[-1]:
            ax.set_xlabel("film thickness (nm)")
        ax.grid(alpha=0.25)
    if wide:
        fig.supxlabel("film thickness (nm)", fontsize=9)
    for ax in (axs[:1] if wide else axs):
        ax.set_ylabel("A vs B change (%)")
    axs[-1 if wide else 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("solid: H = 0 rods, dashed: H ≠ 0 rods, dotted line: commensurate thickness", fontsize=9)


# ----------------------------------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------------------------------
def draw_fit_rod(fig, L, F, sigma, Fc, title="", L_model=None, F_model=None):
    """Data with error bars and model |F| (log scale) above normalized residuals (F - Fc) / sigma."""
    fig.clear()
    a, b = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    a.errorbar(L, F, yerr=sigma, fmt="o", ms=3, color="0.25", ecolor="0.6", elinewidth=0.8, label="data")
    if L_model is not None:
        a.plot(L_model, F_model, color="tab:red", lw=1.2, label="model")
    else:
        a.plot(L, Fc, color="tab:red", lw=1.2, label="model")
    a.set_yscale("log")
    a.set_ylabel("|F| (e)")
    a.set_title(title, fontsize=10)
    a.legend(fontsize=8)
    r = (np.asarray(F) - np.asarray(Fc)) / np.asarray(sigma)
    b.axhline(0, color="k", lw=0.6)
    for y in (-2, 2):
        b.axhline(y, color="0.6", lw=0.6, ls="--")
    b.plot(L, r, "o", ms=3, color="tab:blue")
    b.set_ylabel("(F - Fc) / σ")
    b.set_xlabel("L (r.l.u.)")
    for ax in (a, b):
        ax.grid(alpha=0.25)


def draw_fom_history(fig, history):
    """Figure of merit per DE generation and per least-squares evaluation."""
    fig.clear()
    ax = fig.add_subplot(111)
    for stage, col in (("de", "tab:blue"), ("lsq", "tab:orange")):
        pts = [(h["step"], h["fom"]) for h in history if h["stage"] == stage]
        if pts:
            x, y = zip(*pts)
            ax.semilogy(x, y, marker=".", color=col, label={"de": "differential evolution (best per generation)",
                                                             "lsq": "least squares (per evaluation)"}[stage])
    ax.set_xlabel("generation / evaluation")
    ax.set_ylabel("figure of merit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def draw_correlation(fig, names, corr):
    fig.clear()
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)


def draw_series(fig, x, values, errors, ylabel, xlabel="potential (V)", truth=None, title=""):
    """A per-dataset parameter across a series (e.g. x_OH vs potential) with 1 sigma error bars."""
    fig.clear()
    ax = fig.add_subplot(111)
    ax.errorbar(x, values, yerr=errors, fmt="o-", color="tab:blue", capsize=3, label="fit")
    if truth is not None:
        ax.plot(x, truth, "s--", color="0.5", mfc="none", label="true")
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)


# ----------------------------------------------------------------------------------------------
# in-situ electrochemistry (ctrfit.echem)
# ----------------------------------------------------------------------------------------------
BRANCH_STYLE = {1: dict(color="tab:red", label="anodic"), -1: dict(color="tab:blue", label="cathodic"),
                0: dict(color="0.5", label="hold / no potential")}


def draw_echem_sync(fig, r, ec):
    """Intensity and potential on the potentiostat time axis."""
    fig.clear()
    if r is None or r.t_ec is None:
        message(fig, "Load a SPEC scan and a potentiostat file (or define the CV), then press Update.")
        return
    a, b = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    t0 = r.alignment.ec_t[0]
    x = r.t_ec - t0
    for br, st in BRANCH_STYLE.items():
        m = r.branch == br
        if m.any():
            a.plot(x[m], r.I[m], ".", ms=3, color=st["color"], label=st["label"])
    if r.background is not None and r.background.model != "none":
        a.axvspan(x[0], x[0] + r.background.t_end, color="0.92", zorder=0, label="relaxation window")
    al = r.alignment
    if al.onset_spec is not None:
        a.axvline(al.onset_spec + al.offset - t0, color="tab:green", lw=1, label="intensity onset")
    a.set_ylabel("intensity")
    a.legend(fontsize=7, loc="best")
    b.plot(al.ec_t - t0, ec.potential, color="k", lw=0.8, label="potentiostat")
    ok = np.isfinite(r.V)
    b.plot(x[ok], r.V[ok], ".", ms=2, color="tab:orange", label="at the SPEC points")
    if al.sweep_start_ec is not None:
        b.axvline(al.sweep_start_ec - t0, color="tab:green", lw=1, label="sweep start")
    b.set_ylabel("potential (V)")
    b.set_xlabel("time from the potentiostat start (s)")
    b.legend(fontsize=7, loc="best")
    a.set_title(f"{al.mode} sync, offset {al.offset:+.2f} s", fontsize=10)
    for ax in (a, b):
        ax.grid(alpha=0.25)


def draw_echem_background(fig, r):
    fig.clear()
    if r is None or r.background is None:
        message(fig, "Press Update to fit the relaxation-period background.")
        return
    a, b = fig.subplots(2, 1, sharex=True)
    bg = r.background
    a.plot(r.t, r.I, ".", ms=3, color="0.4", label="intensity")
    if bg.model != "none":
        inside = r.t <= bg.t_end
        a.plot(r.t[inside], bg(r.t[inside]), color="tab:red", lw=1.5, label=f"{bg.model} fit")
        a.plot(r.t[~inside], bg(r.t[~inside]), color="tab:red", lw=1, ls="--", label="extrapolated")
        a.axvspan(r.t[0], bg.t_end, color="0.92", zorder=0)
    a.set_ylabel("intensity")
    a.legend(fontsize=7)
    b.plot(r.t, r.I_corr, ".", ms=3, color="k")
    b.set_ylabel(f"corrected ({bg.correction})")
    b.set_xlabel("time from the first SPEC point (s)")
    for ax in (a, b):
        ax.grid(alpha=0.25)


def draw_echem_iv(fig, r, cycles=None, show_points=True):
    """Corrected intensity vs potential: points by cycle, bins (or points) per sweep direction, fits."""
    fig.clear()
    if r is None or r.I_corr is None:
        message(fig, "Press Update to see intensity vs potential.")
        return
    ax = fig.add_subplot(111)
    if show_points:
        cyc = sorted({int(c) for c in r.cycle if c > 0} if cycles is None else cycles)
        cmap = cm.viridis
        for k, c in enumerate(cyc):
            col = cmap(k / max(len(cyc) - 1, 1))
            for br, mk in ((1, "o"), (-1, "s")):
                m = (r.cycle == c) & (r.branch == br) & np.isfinite(r.V)
                if m.any():
                    ax.plot(r.V[m], r.I_corr[m], mk, ms=2.5, color=col, alpha=0.35,
                            mfc=col if br > 0 else "none", label=f"cycle {c}" if br > 0 else None)
    for b in r.binned:
        st = BRANCH_STYLE[b.branch]
        ax.errorbar(b.V, b.I, yerr=b.sigma, fmt="o-" if b.V.size < 400 else "-", ms=3, lw=1, color=st["color"],
                    label=f"{st['label']} (binned)", capsize=0, elinewidth=0.6)
        f = r.fits.get(b.branch)
        if f is not None:
            v = np.linspace(b.V.min(), b.V.max(), 400)
            ax.plot(v, f.model(v), color=st["color"], lw=2, alpha=0.6, ls="--")
            for t in f.transitions:
                ax.axvline(t.E0, color=st["color"], lw=0.8, ls=":")
    ax.set_xlabel("potential (V)")
    ax.set_ylabel("corrected intensity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.set_title("filled: anodic, open: cathodic; dashed: sigmoid fits", fontsize=9)


def draw_echem_fit(fig, r):
    fig.clear()
    if r is None or not r.fits:
        message(fig, "Press 'Fit transitions' (needs enough points per sweep direction).")
        return
    axs = fig.subplots(2, len(r.fits), sharex="col", squeeze=False, gridspec_kw=dict(height_ratios=[3, 1]))
    for j, (br, f) in enumerate(sorted(r.fits.items(), reverse=True)):
        a, b = axs[0, j], axs[1, j]
        st = BRANCH_STYLE[br]
        a.errorbar(f.V, f.I, yerr=f.sigma, fmt="o", ms=3, color=st["color"], elinewidth=0.6)
        v = np.linspace(f.V.min(), f.V.max(), 400)
        base, comps = f.model(v, parts=True)
        a.plot(v, f.model(v), color="k", lw=1.5, label="fit")
        for k, (c, t) in enumerate(zip(comps, f.transitions)):
            a.plot(v, base + c, lw=0.8, ls="--", label=f"E{k + 1} = {t.E0:.3f} V, {1e3 * t.width:.0f} mV")
        a.set_title(f"{st['label']} sweep, reduced chi2 {f.red_chi2:.2f}", fontsize=10)
        a.legend(fontsize=7)
        b.plot(f.V, (f.I - f.model(f.V)) / f.sigma, ".", color=st["color"])
        b.axhline(0, color="k", lw=0.6)
        b.set_ylabel("residual / σ")
        b.set_xlabel("potential (V)")
        a.set_ylabel("intensity")
        for ax in (a, b):
            ax.grid(alpha=0.25)


def draw_echem_model(fig, r, pm=None):
    """Predicted intensity vs potential for the composition path (relative to the reference state),
    the data on the same scale, the path fit, and the fractions (path and from the data)."""
    fig.clear()
    if r is None or not r.sim:
        message(fig, "Enter the composition path and press 'Simulate'.")
        return
    sim = r.sim
    a, b = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    ref = sim["ref_model"]
    a.plot(sim["V"], sim["anodic"] / ref, color="tab:red", lw=1.5, label="model, anodic")
    if not np.allclose(sim["anodic"], sim["cathodic"]):
        a.plot(sim["V"], sim["cathodic"] / ref, color="tab:blue", lw=1.5, label="model, cathodic")
    if r.I_corr is not None and r.scale:
        for br in (1, -1):
            m = (r.branch == br) & np.isfinite(r.V)
            if m.any():
                a.plot(r.V[m], r.I_corr[m] / r.scale / ref, ".", ms=2, alpha=0.35,
                       color=BRANCH_STYLE[br]["color"], label=f"data ({BRANCH_STYLE[br]['label']})")
    if r.path_fit is not None and pm is not None:
        p = r.path_fit
        for br, ls in ((1, "-"), (-1, "--")):
            a.plot(sim["V"], p.curve(pm, sim["V"], np.full(sim["V"].size, br)) / (r.scale or p.scale) / ref,
                   color="k", lw=1, ls=ls, label="path fit" if br > 0 else None)
    a.set_ylabel("I / I(reference state)")
    a.legend(fontsize=7, ncol=2)
    states = " → ".join(sim["states"])
    a.set_title(f"CTR model at the HKL, path {states}", fontsize=10)
    names = ("H2O", "OH", "O")
    cols = ("tab:cyan", "tab:orange", "tab:purple")
    for i, (nm, c) in enumerate(zip(names, cols)):
        if np.any(sim["fractions"][:, i] > 0):
            b.plot(sim["V"], sim["fractions"][:, i], color=c, lw=1.5, label=f"{nm} (path)")
    if r.coverage is not None:
        ok = np.isfinite(r.V)
        for i, (nm, c) in enumerate(zip(names, cols)):
            if np.any(r.coverage.fractions[ok, i] > 0):
                b.plot(r.V[ok], r.coverage.fractions[ok, i], ".", ms=2, color=c, alpha=0.4, label=f"{nm} (from data)")
    b.set_ylabel("fraction of CUS sites")
    b.set_xlabel("potential (V)")
    b.set_ylim(-0.05, 1.05)
    b.legend(fontsize=7, ncol=2)
    for ax in (a, b):
        ax.grid(alpha=0.25)
