"""HDF5 metadata helpers (from sxrd_meta_viewer, no Qt). h5py is optional."""
from ctrfit.datatools.images import parse_filename


try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

def scan_h5_dir_with_masters(dirpath):
    """Like sxrd_bgsub.scan_directory but KEEPS master files.
    Returns list of file info dicts with an is_master flag."""
    import os as _os
    entries = []
    for f in sorted(_os.listdir(dirpath)):
        full = _os.path.join(dirpath, f)
        if not _os.path.isfile(full):
            continue
        if parse_filename is not None:
            info = parse_filename(full)
            if info is not None:
                info['is_master'] = 'master' in info['suffix']
                if info['ext'] in ('h5', 'nxs', 'hdf', 'hdf5'):
                    entries.append(info)
                continue
        # Fallback without bgsub: keep any .h5 file
        name = f.lower()
        if name.endswith(('.h5', '.nxs', '.hdf', '.hdf5')):
            entries.append({
                'scan': 0, 'image': 0, 'path': full,
                'ext': name.rsplit('.', 1)[-1], 'prefix': '', 'suffix': '',
                'is_master': 'master' in name,
            })
    # Sort: by scan then master-first then image
    entries.sort(key=lambda x: (x['scan'], 0 if x.get('is_master') else 1,
                                 x['image']))
    return entries


def describe_h5(obj):
    if isinstance(obj, h5py.Dataset):
        info = f"dataset  shape={obj.shape}  dtype={obj.dtype}"
        if obj.size <= 10 and obj.ndim <= 1:
            try:
                val = obj[...]
                if obj.dtype.kind in ('S', 'O', 'U'):
                    s = str(val)
                    if len(s) > 80: s = s[:77] + "..."
                    info += f"  = {s}"
                else:
                    info += f"  = {val}"
            except Exception: pass
        elif obj.size == 1:
            try: info += f"  = {obj[()]}"
            except Exception: pass
        return info
    elif isinstance(obj, h5py.Group):
        return f"group  ({len(obj)} children)"
    return "unknown"
