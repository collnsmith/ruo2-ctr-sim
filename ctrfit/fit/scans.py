"""Scans of the loaded data files: which are fitted, and which are fitted together.

A loaded file (a 'source', e.g. one film at one potential from the scan library's export) holds the
points of several scans (its 'scan' column; a file without one counts one scan per rod). What a fit
sees are groups: each group of scans is one fitted dataset, with its own scale and its own copy of
the per-dataset parameters (what the potential changes), and the scans of one group share one
surface structure. By default each file is one group, so the fitted datasets are the files.

    sel = ScanSelection()
    sel.add(Dataset.load("film7_CA_0.5V.csv"))
    sel.add(Dataset.load("film7_CA_0.5V_repeat.csv"))
    sel.set_use([sel.key("film7 CA 0.5 V", "#333")], False)        # leave a scan out
    sel.set_group(sel.keys_of("film7 CA 0.5 V repeat"), "film7 CA 0.5 V")   # fit the repeat with it
    datasets, notes = regroup(model, sel, previous=datasets)

Points can also be cut: L windows per file and rod (sel.add_cut("film7 CA 0.5 V", "0 1", 2.9, 3.1)) leave
those points of every scan of that rod in that file out of the fitted datasets; the files stay as loaded.

Group names are free text: scans given the same group are fitted together, from one file or several,
and a file's scans can be split into several groups. A group that mixes films or potentials is
allowed (e.g. to test whether the potential changes anything) and noted: it gets one surface
structure, and a mixed film is its own film (its own thickness) in a fit per film.

regroup() also keeps the parameters in step: a group keeps its scope (and so its parameter copies,
scale and error floor) while its name stays; a new group starts from the copies of the dataset its
first scan was fitted in; the copies of a dataset that is gone are removed (when none of a
parameter's copies are left, the global parameter takes the first one's value and fit flag back).
"""
from dataclasses import dataclass

import numpy as np

from ..data.dataset import Dataset

SEP = "\t"          # key = source name + SEP + scan label


@dataclass
class ScanRow:
    key: str
    source: str            # the loaded file's dataset name
    scan: str
    rods: list
    n: int
    L_min: float
    L_max: float
    film: str = None
    thickness_nm: float = None
    potential_V: float = None
    condition: str = None
    use: bool = True
    group: str = ""        # the group it is fitted in (the source's name unless set)
    custom: bool = False   # group set by the user
    n_cut: int = 0         # its points inside a cut window (left out of the fit)


def _unique(name, taken):
    if name not in taken:
        return name
    i = 2
    while f"{name} ({i})" in taken:
        i += 1
    return f"{name} ({i})"


class ScanSelection:
    def __init__(self, sources=(), choices=None):
        self.sources = []
        self.choices = {}          # key -> {"use": bool, "group": str} (only what differs from the default)
        self.cuts = {}             # source name -> {rod label: [[L_lo, L_hi], ...]} (points left out)
        self.placed = {}           # key -> name of the dataset it went to in the last build
        for ds in sources:
            self.add(ds)
        for k, v in (choices or {}).items():
            self.choices[k] = dict(v)

    @classmethod
    def from_datasets(cls, datasets):
        """A selection whose default build gives these datasets (each one file, every scan used)."""
        sel = cls()
        for ds in datasets:
            src = sel.add(ds.copy())
            ds.name = src.name
            sel.placed.update({sel.key(src.name, s): src.name for s in src.scan_labels})
        return sel

    # ------------------------------------------------------------------ sources
    @staticmethod
    def key(source, scan):
        return f"{source}{SEP}{scan}"

    def source(self, name):
        return next((s for s in self.sources if s.name == name), None)

    def add(self, ds):
        """Add a loaded file; a name already loaded gets ' (2)', ' (3)', ... Returns the source."""
        ds.name = _unique(ds.name, {s.name for s in self.sources})
        self.sources.append(ds)
        return ds

    def remove(self, names):
        names = set(names)
        self.sources = [s for s in self.sources if s.name not in names]
        for k in [k for k in self.choices if k.split(SEP, 1)[0] in names]:
            del self.choices[k]
        for n in names:
            self.cuts.pop(n, None)

    def clear(self):
        self.sources, self.choices, self.placed, self.cuts = [], {}, {}, {}

    # ------------------------------------------------------------------ choices
    def rows(self):
        out = []
        for src in self.sources:
            meta = src.meta
            cut = self.cut_mask(src)
            for scan, idx in src.scans().items():
                k = self.key(src.name, scan)
                c = self.choices.get(k, {})
                group = c.get("group") or src.name
                L = src.L[idx]
                out.append(ScanRow(k, src.name, scan, list(dict.fromkeys(src.rod[idx].tolist())), int(idx.size),
                                   float(L.min()), float(L.max()), meta.get("film"), meta.get("thickness_nm"),
                                   meta.get("potential_V"), meta.get("condition"), bool(c.get("use", True)),
                                   group, group != src.name, int(cut[idx].sum())))
        return out

    def keys_of(self, source):
        return [r.key for r in self.rows() if r.source == source]

    def _set(self, key, field, value, default):
        src = key.split(SEP, 1)[0]
        if self.source(src) is None:
            raise KeyError(f"no loaded scan {key!r}")
        c = self.choices.setdefault(key, {})
        if value == default:
            c.pop(field, None)
        else:
            c[field] = value
        if not c:
            del self.choices[key]

    def set_use(self, keys, use=True):
        for k in keys:
            self._set(k, "use", bool(use), True)

    def set_group(self, keys, group=""):
        """Fit these scans together in `group` ('' or the file's own name: back to their file's group)."""
        group = (group or "").strip()
        for k in keys:
            src = k.split(SEP, 1)[0]
            self._set(k, "group", "" if group == src else group, "")

    def reset(self):
        self.choices = {}

    # ------------------------------------------------------------------ cuts: L windows left out per file and rod
    def cuts_of(self, source, rod):
        """The cut windows [(L_lo, L_hi)] of rod `rod` (a label such as '0 1') in file `source`."""
        return [tuple(w) for w in self.cuts.get(source, {}).get(rod, [])]

    def add_cut(self, source, rod, lo, hi):
        """Leave out the points of rod `rod` in file `source` with lo <= L <= hi, in every scan of that rod in
        the file. Windows that overlap or touch are merged. Returns the rod's windows."""
        if self.source(source) is None:
            raise KeyError(f"no loaded file {source!r}")
        lo, hi = sorted((float(lo), float(hi)))
        if not hi > lo:
            raise ValueError("a cut window needs two different L")
        merged = []
        for a, b in sorted(self.cuts_of(source, rod) + [(lo, hi)]):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self.cuts.setdefault(source, {})[rod] = [list(w) for w in merged]
        return merged

    def remove_cut(self, source, rod, L=None):
        """Remove the window of this rod that holds L (L None: all its windows). True if one was removed."""
        ws = self.cuts_of(source, rod)
        keep = [] if L is None else [w for w in ws if not w[0] <= L <= w[1]]
        if len(keep) == len(ws):
            return False
        if keep:
            self.cuts[source][rod] = [list(w) for w in keep]
        else:
            self.cuts[source].pop(rod, None)
            if not self.cuts[source]:
                del self.cuts[source]
        return True

    def clear_cuts(self, source=None):
        """Remove every cut of a file (None: of every file)."""
        if source is None:
            self.cuts = {}
        else:
            self.cuts.pop(source, None)

    def cut_mask(self, src):
        """True for the points of a loaded file (Dataset) inside one of its cut windows (ends included)."""
        mask = np.zeros(len(src), bool)
        for rod, ws in self.cuts.get(src.name, {}).items():
            on = src.rod == rod
            for lo, hi in ws:
                mask |= on & (src.L >= lo - 1e-9) & (src.L <= hi + 1e-9)
        return mask

    def _uncut(self, src):
        mask = self.cut_mask(src)
        return src.subset(~mask) if mask.any() else src

    def cut_points(self, name):
        """The cut points of the scans fitted in group `name` (for plotting), or None."""
        parts = []
        for r in self.rows():
            if r.use and r.group == name and r.n_cut:
                src = self.source(r.source)
                parts.append(src.subset((src.point_scans() == r.scan) & self.cut_mask(src)))
        if not parts:
            return None
        cat = lambda a: np.concatenate([getattr(p, a) for p in parts])
        return Dataset(cat("H"), cat("K"), cat("L"), cat("F"), cat("sigma"), cat("rod"), name=f"{name} (cut)",
                       sys_floor=parts[0].sys_floor, scan=cat("scan"))

    # ------------------------------------------------------------------ building
    def build(self, previous=()):
        """([Dataset] one per group with used scans, notes). A group keeps the scope and error floor of
        the previous dataset of its name. Each dataset's .parts lists its scan keys."""
        prev = {d.name: d for d in previous}
        groups = {}
        for r in self.rows():
            if r.use:
                groups.setdefault(r.group, []).append(r)
        out, notes, placed = [], [], {}
        for name, rows in groups.items():
            ds, n = self._merge(name, rows)
            notes += n
            if not len(ds):
                notes.append(f"group {name}: every point is cut; it is not fitted")
                continue
            old = prev.get(name)
            ds.scope = old.scope if old is not None else ""
            if old is not None:
                ds.sys_floor = old.sys_floor
            ds.parts = [r.key for r in rows]
            placed.update({r.key: name for r in rows})
            out.append(ds)
        self.placed = placed
        return out, notes

    def _merge(self, name, rows):
        by_src = {}
        for r in rows:
            by_src.setdefault(r.source, []).append(r.scan)
        srcs = [self._uncut(self.source(s)) for s in by_src]        # without the points of cut windows
        whole = len(srcs) == 1 and srcs[0].name == name and len(by_src[name]) == len(srcs[0].scan_labels)
        if whole:
            ds = srcs[0].copy()
            ds.scan = ds.point_scans()
            return ds, []
        labels = [r.scan for r in rows]
        clash = {s for s in labels if labels.count(s) > 1}
        parts = []
        for src in srcs:
            lab = src.point_scans()
            keep = np.isin(lab, by_src[src.name])
            sub = src.subset(keep)
            sub.scan = np.array([f"{src.name}: {s}" if s in clash else s for s in lab[keep]], dtype=str)
            parts.append(sub)
        meta, notes = _merge_meta(name, srcs)
        cat = lambda a: np.concatenate([getattr(p, a) for p in parts])
        ds = Dataset(cat("H"), cat("K"), cat("L"), cat("F"), cat("sigma"), cat("rod"), name=name,
                     sys_floor=srcs[0].sys_floor, scan=cat("scan"), **meta)
        return ds, notes

    def left_out(self, name):
        """The points of the scans left out of group `name` (for plotting), or None."""
        parts = []
        for r in self.rows():
            if not r.use and r.group == name:
                src = self.source(r.source)
                parts.append(src.subset(src.point_scans() == r.scan))
        if not parts:
            return None
        cat = lambda a: np.concatenate([getattr(p, a) for p in parts])
        return Dataset(cat("H"), cat("K"), cat("L"), cat("F"), cat("sigma"), cat("rod"), name=f"{name} (left out)",
                       sys_floor=parts[0].sys_floor, scan=cat("scan"))

    def summary(self):
        rows = self.rows()
        used = [r for r in rows if r.use]
        return (f"{len(used)} of {len(rows)} scans fitted, in {len({r.group for r in used})} dataset(s)"
                if rows else "No data loaded")

    # ------------------------------------------------------------------ io
    def to_dict(self):
        return dict(sources=[s.to_dict() for s in self.sources],          # copies: a snapshot stays as it was
                    choices={k: dict(v) for k, v in self.choices.items()},
                    cuts={src: {rod: [list(w) for w in ws] for rod, ws in rods.items()} for src, rods in self.cuts.items()})

    @classmethod
    def from_dict(cls, d):
        sel = cls()
        sel.sources = [Dataset.from_dict(x) for x in d.get("sources", [])]
        sel.choices = {k: dict(v) for k, v in d.get("choices", {}).items()}
        sel.cuts = {src: {rod: [[float(a), float(b)] for a, b in ws] for rod, ws in rods.items()}
                    for src, rods in d.get("cuts", {}).items()}
        return sel


def _merge_meta(name, srcs):
    """Metadata of a group from several files: kept where they agree, noted where they do not."""
    notes, meta = [], {}
    get = lambda k: [s.meta.get(k) for s in srcs]
    films = list(dict.fromkeys(f for f in get("film") if f))
    thick = list(dict.fromkeys(t for t in get("thickness_nm") if t is not None))
    pots = get("potential_V")
    conds = list(dict.fromkeys(c for c in get("condition") if c))
    energy = list(dict.fromkeys(e for e in get("energy_kev") if e is not None))
    if films:
        meta["film"] = " + ".join(films)
    if len(films) > 1:
        notes.append(f"group {name}: films {', '.join(films)} together: one surface structure, and in a fit per "
                     "film it is a film of its own")
    if len(thick) == 1:
        meta["thickness_nm"] = thick[0]
    elif len(thick) > 1:
        notes.append(f"group {name}: thicknesses {', '.join(f'{t:g}' for t in thick)} nm: no nominal thickness")
    if len(set(pots)) == 1 and pots[0] is not None:
        meta["potential_V"] = pots[0]
    elif len(set(pots)) > 1 or len(conds) > 1:
        what = ", ".join(conds) if len(conds) > 1 else "different potentials"
        notes.append(f"group {name}: {what} together: one surface structure for all of them")
    if conds:
        meta["condition"] = " + ".join(conds)
    if energy:
        meta["energy_kev"] = energy[0]
        if len(energy) > 1:
            notes.append(f"group {name}: energies {', '.join(f'{e:g}' for e in energy)} keV: the first is used")
    meta["notes"] = "scans of " + ", ".join(s.name for s in srcs)
    return meta, notes


# ---------------------------------------------------------------------- parameters
def _free_scope(taken):
    i = 1
    while f"ds{i}" in taken:
        i += 1
    return f"ds{i}"


def prune_scopes(model, scopes, order=()):
    """Remove the parameter copies of dataset scopes that are gone. A link that pointed at a removed
    copy is cut (the parameter keeps its value). When no copy of a parameter is left, the global one
    takes the value and fit flag of the first removed copy (in `order` of scopes), so fitting it goes
    on as before. Returns the removed names."""
    P = model.params
    scopes = set(scopes)
    if not scopes:
        return []
    vals = P.values()
    gone = [p for p in P if p.scope in scopes]
    rank = {s: i for i, s in enumerate(order)}
    gone.sort(key=lambda p: rank.get(p.scope, len(rank)))
    for p in gone:
        P.remove(p.name)
    for p in list(P):
        if p.linked:
            try:
                for ref in p._expr.names:
                    P.resolve(ref, p.scope)
            except Exception:
                p.value = vals[p.name]
                p.expr, p._expr = None, None
    left = {p.base for p in P if p.scope}
    done = set()
    for p in gone:
        b = p.base
        if b in done or b in left or b not in P:
            continue
        done.add(b)
        g = P[b]
        if g.linked or g.fit:
            continue
        g.value = float(np.clip(vals[p.name], g.min, g.max))
        g.fit = p.fit
    return [p.name for p in gone]


def regroup(model, selection, previous=()):
    """The fitted datasets for the selection's current choices, with the model's parameters kept in
    step (see the module notes). Returns (datasets, notes)."""
    previous = list(previous)
    placed = dict(selection.placed)
    by_name = {d.name: d for d in previous}
    datasets, notes = selection.build(previous)
    old_scopes = [d.scope for d in previous if d.scope]
    kept = {d.scope for d in datasets if d.scope}
    P = model.params
    snapshot = {}
    vals = P.values()
    for s in old_scopes:
        snapshot[s] = [(p, vals[p.name]) for p in P if p.scope == s]
    prune_scopes(model, [s for s in old_scopes if s not in kept], order=old_scopes)
    if len(datasets) > 1:
        taken = set(kept)
        for ds in datasets:
            if not ds.scope:
                ds.scope = _free_scope(taken)
                taken.add(ds.scope)
    for ds in datasets:                 # a new scope starts from the copies of its first scan's dataset
        if not ds.scope or ds.scope in kept:
            continue
        origin = next((by_name[placed[k]] for k in ds.parts if placed.get(k) in by_name), None)
        if origin is None or not origin.scope:
            continue
        tail = f"({origin.name})"
        for p, v in snapshot.get(origin.scope, []):
            name = f"{ds.scope}.{p.base}"
            if name in P or p.base not in P:
                continue
            desc = p.description[:-len(tail)] + f"({ds.name})" if p.description.endswith(tail) else p.description
            P.add(name=name, value=float(np.clip(v, p.min, p.max)), min=p.min, max=p.max, fit=p.fit, unit=p.unit,
                  group=p.group, description=desc)
    from .global_fit import complete_per_film
    notes += complete_per_film(model, datasets)    # e.g. a new file of a film whose datasets share copies
    return datasets, notes
