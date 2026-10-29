"""Final figures from the plotter (no Qt).

The plotter's working view stays as it is; Export figure takes a snapshot of what it shows
(FigureSpec: the traces of the main axis and of a second y axis, labels, scales, limits) and draws
it again with a FigureStyle (size, fonts, colours, ticks, grid, legend, per-trace look), so the
saved image is styled without touching the working view.

    spec = FigureSpec.from_axes(fig.axes)      # snapshot of the working plot
    style = FigureStyle.from_spec(spec)        # defaults seeded from it (labels, trace colours)
    render(figure, spec, style)                # draw on any matplotlib Figure
    save(spec, style, path)                    # write PNG / SVG / PDF / EPS / TIFF / JPG
"""
import json
from dataclasses import asdict, dataclass, field, fields

import numpy as np
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

FORMATS = ("png", "svg", "pdf", "eps", "tif", "jpg")
LINESTYLES = ("-", "--", "-.", ":", "none")
MARKERS = ("none", "o", "s", "^", "v", "D", "x", "+", ".", "*")
LEGEND_LOCS = ("best", "upper right", "upper left", "lower left", "lower right", "right", "center left",
               "center right", "lower center", "upper center", "center", "outside right")
FONTS = ("DejaVu Sans", "sans-serif", "serif", "monospace", "Arial", "Helvetica", "Times New Roman")
CM = 2.54


@dataclass
class TraceSpec:
    x: np.ndarray
    y: np.ndarray
    label: str = ""
    color: str = "#1f77b4"
    linestyle: str = "-"
    marker: str = "none"
    axis: str = "left"                 # 'left' or 'right' (second y axis)


@dataclass
class FigureSpec:
    traces: list
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    y2label: str = ""
    xlim: tuple = None
    ylim: tuple = None
    y2lim: tuple = None
    xlog: bool = False
    ylog: bool = False

    @property
    def has_right(self):
        return any(t.axis == "right" for t in self.traces)

    @classmethod
    def from_axes(cls, axes):
        """Snapshot of a working plot: the lines of axes[0] (left) and of a twin axis (right)."""
        axes = list(axes)
        if not axes:
            raise ValueError("nothing is plotted")
        main = axes[0]
        traces = []
        for ax, side in [(main, "left")] + [(a, "right") for a in axes[1:]]:
            for ln in ax.get_lines():
                x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
                if x.size == 0:
                    continue
                label = ln.get_label()
                if label.startswith("_"):
                    label = ""
                ls = ln.get_linestyle()
                ls = {"None": "none", "": "none", " ": "none"}.get(ls, ls)
                mk = ln.get_marker()
                mk = "none" if mk in (None, "None", "", " ") else str(mk)
                traces.append(TraceSpec(x, y, label, to_hex(ln.get_color()), ls if ls in LINESTYLES else "-",
                                        mk if mk in MARKERS else "o", side))
        if not traces:
            raise ValueError("nothing is plotted")
        right = axes[1] if len(axes) > 1 else None
        return cls(traces, main.get_title(), main.get_xlabel(), main.get_ylabel(),
                   right.get_ylabel() if right is not None else "",
                   tuple(main.get_xlim()), tuple(main.get_ylim()),
                   tuple(right.get_ylim()) if right is not None else None,
                   main.get_xscale() == "log", main.get_yscale() == "log")


@dataclass
class TraceStyle:
    label: str
    color: str
    visible: bool = True
    linestyle: str = "-"
    linewidth: float = 1.5
    marker: str = "none"
    markersize: float = 4.0
    axis: str = "left"
    source: str = ""                   # the trace's label in the working plot (labels can be edited)


@dataclass
class FigureStyle:
    width: float = 6.0                 # inches (the window shows cm too)
    height: float = 4.5
    dpi: int = 300
    font: str = "DejaVu Sans"
    title_size: float = 12
    label_size: float = 11
    tick_size: float = 9
    legend_size: float = 9
    fig_bg: str = "#ffffff"
    axes_bg: str = "#ffffff"
    text_color: str = "#000000"
    transparent: bool = False
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    y2label: str = ""
    xmin: str = ""                     # '' = automatic
    xmax: str = ""
    ymin: str = ""
    ymax: str = ""
    y2min: str = ""
    y2max: str = ""
    xlog: bool = False
    ylog: bool = False
    grid: bool = False
    grid_style: str = ":"
    grid_alpha: float = 0.5
    minor_ticks: bool = True
    ticks_in: bool = True
    ticks_all_sides: bool = True
    spine_width: float = 1.0
    legend: bool = True
    legend_loc: str = "best"
    legend_frame: bool = False
    tight: bool = True
    traces: list = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec, base=None, keep_text=False, keep_traces=False):
        """A style for this plot. Size, fonts, colours, ticks, grid and legend come from `base` (e.g. the
        last export) so a series of figures keeps one look; title, labels, scales and limits from the
        spec unless keep_text; each trace from the plot, or with keep_traces its look from the base
        trace of the same label (a refresh keeps the edits)."""
        st = cls(**{f.name: getattr(base, f.name) for f in fields(cls) if f.name != "traces"}) if base else cls()
        if not (keep_text and base):
            st.title, st.xlabel, st.ylabel, st.y2label = spec.title, spec.xlabel, spec.ylabel, spec.y2label
            st.xlog, st.ylog = spec.xlog, spec.ylog
            st.xmin = st.xmax = st.ymin = st.ymax = st.y2min = st.y2max = ""
        old = {(t.source or t.label): t for t in (base.traces if base else [])}
        st.traces = []
        for t in spec.traces:
            o = old.get(t.label)
            if keep_traces and o is not None:
                st.traces.append(TraceStyle(**asdict(o)))
            else:
                st.traces.append(TraceStyle(t.label, t.color, True, t.linestyle, o.linewidth if o else 1.5,
                                            t.marker, o.markersize if o else 4.0, t.axis, t.label))
        return st

    def to_json(self):
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        traces = [TraceStyle(**t) for t in d.pop("traces", [])]
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known}, traces=traces)


def _lim(a, b):
    def num(s):
        s = str(s).strip()
        if not s or s.lower() == "auto":
            return None
        return float(s)
    return num(a), num(b)


def render(fig, spec, style):
    """Draw the spec on a matplotlib Figure with the style (the figure is cleared first)."""
    fig.clear()
    fig.set_facecolor(style.fig_bg)
    tc = style.text_color
    rc = {"font.family": style.font}
    import matplotlib
    with matplotlib.rc_context(rc):
        ax = fig.add_subplot(111)
        ax2 = ax.twinx() if any(t.axis == "right" and t.visible for t in style.traces) else None
        for a in (ax, ax2):
            if a is None:
                continue
            a.set_facecolor(style.axes_bg if a is ax else "none")
            for sp in a.spines.values():
                sp.set_color(tc)
                sp.set_linewidth(style.spine_width)
            a.tick_params(colors=tc, labelsize=style.tick_size, direction="in" if style.ticks_in else "out",
                          width=style.spine_width, which="both")
            for lab in a.get_xticklabels() + a.get_yticklabels():
                lab.set_fontfamily(style.font)
        handles = []
        for t, ts in zip(spec.traces, style.traces):
            if not ts.visible:
                continue
            target = ax2 if (ts.axis == "right" and ax2 is not None) else ax
            ls = "None" if ts.linestyle == "none" else ts.linestyle
            mk = None if ts.marker == "none" else ts.marker
            if ls == "None" and mk is None:
                mk = "o"
            h, = target.plot(t.x, t.y, color=ts.color, linestyle=ls, linewidth=ts.linewidth, marker=mk,
                             markersize=ts.markersize, label=ts.label or None)
            if ts.label:
                handles.append(h)
        font = {"family": style.font, "color": tc}
        if style.title:
            ax.set_title(style.title, fontsize=style.title_size, fontdict=font)
        ax.set_xlabel(style.xlabel, fontsize=style.label_size, fontdict=font)
        ax.set_ylabel(style.ylabel, fontsize=style.label_size, fontdict=font)
        if style.xlog:
            ax.set_xscale("log")
        if style.ylog:
            ax.set_yscale("log")
        lo, hi = _lim(style.xmin, style.xmax)
        if lo is not None or hi is not None:
            ax.set_xlim(left=lo, right=hi)
        lo, hi = _lim(style.ymin, style.ymax)
        if lo is not None or hi is not None:
            ax.set_ylim(bottom=lo, top=hi)
        if ax2 is not None:
            ax2.set_ylabel(style.y2label, fontsize=style.label_size, fontdict=font)
            lo, hi = _lim(style.y2min, style.y2max)
            if lo is not None or hi is not None:
                ax2.set_ylim(bottom=lo, top=hi)
        if style.minor_ticks:
            if not style.xlog:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
            if not style.ylog:
                ax.yaxis.set_minor_locator(AutoMinorLocator())
            if ax2 is not None:
                ax2.yaxis.set_minor_locator(AutoMinorLocator())
        if style.ticks_all_sides:
            ax.tick_params(top=True, which="both")
            if ax2 is None:
                ax.tick_params(right=True, which="both")
        if style.grid:
            ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha, color=tc)
        if style.legend and handles:
            kw = dict(fontsize=style.legend_size, frameon=style.legend_frame, labelcolor=tc,
                      prop={"family": style.font, "size": style.legend_size})
            kw.pop("fontsize")
            top = ax2 if ax2 is not None else ax          # drawn last, so no line of either axis covers it
            if style.legend_loc == "outside right":
                leg = top.legend(handles=handles, loc="center left", bbox_to_anchor=(1.12 if ax2 else 1.02, 0.5), **kw)
            else:
                leg = top.legend(handles=handles, loc=style.legend_loc, **kw)
            if style.legend_frame:
                leg.get_frame().set_facecolor(style.axes_bg)
                leg.get_frame().set_edgecolor(tc)
    if style.tight:
        try:
            fig.tight_layout()
        except Exception:      # tight_layout can fail on odd sizes; the plain layout is still usable
            pass
    return fig


def save(spec, style, path, fmt=None):
    """Render at the style's size and write the file (format from the extension if not given)."""
    fmt = (fmt or str(path).rsplit(".", 1)[-1]).lower()
    fmt = {"tiff": "tif", "jpeg": "jpg"}.get(fmt, fmt)
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    fig = Figure(figsize=(style.width, style.height), dpi=style.dpi)
    render(fig, spec, style)
    kw = dict(dpi=style.dpi, facecolor=fig.get_facecolor(), transparent=style.transparent)
    if style.legend and style.legend_loc == "outside right":
        kw["bbox_inches"] = "tight"
    if fmt == "tif":
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(path, format={"tif": "tiff", "jpg": "jpeg"}.get(fmt, fmt), **kw)
    return path
