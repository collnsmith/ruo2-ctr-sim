"""Write a fit result to disk: JSON, text summary and PNG plots (headless, matplotlib Agg)."""
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from . import plots


def _fig(w, h):
    try:
        fig = Figure(figsize=(w, h), layout="constrained")
    except TypeError:  # matplotlib < 3.5
        fig = Figure(figsize=(w, h), constrained_layout=True)
    FigureCanvasAgg(fig)
    return fig


def write_report(fit, result, out_dir, dense_step=0.01):
    """result.json, summary.txt, one PNG per rod (data, model, residuals), fom_history.png and
    correlation.png. Returns the list of files written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [out / "result.json", out / "summary.txt"]
    result.save(files[0])
    files[1].write_text(result.summary + "\n", encoding="utf-8")
    for ds, ev in zip(fit.datasets, fit.evaluators):
        Fc = ev(fit.model.params.view(ds.scope))
        for lab, idx in ds.rods().items():
            H, K = ds.rod_hk(lab)
            Lr = ds.L[idx]
            Ld = np.arange(Lr.min(), Lr.max() + 1e-9, dense_step)
            Fd = fit.model.evaluator(np.full(Ld.size, H), np.full(Ld.size, K), Ld, scope=ds.scope)()
            fig = _fig(7, 5)
            plots.draw_fit_rod(fig, Lr, ds.F[idx], ds.sigma_eff[idx], Fc[idx], f"{ds.name}: ({H} {K} L)", Ld, Fd)
            p = out / f"rod_{ds.name}_{H}_{K}.png".replace(" ", "_")
            fig.savefig(p, dpi=110)
            files.append(p)
    fig = _fig(7, 4)
    plots.draw_fom_history(fig, result.history)
    fig.savefig(out / "fom_history.png", dpi=110)
    files.append(out / "fom_history.png")
    if len(result.names) > 1:
        fig = _fig(6, 5)
        plots.draw_correlation(fig, result.names, result.correlation)
        fig.savefig(out / "correlation.png", dpi=110)
        files.append(out / "correlation.png")
    return files
