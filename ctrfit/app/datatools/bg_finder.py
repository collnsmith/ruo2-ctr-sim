"""SXRD Background ROI Finder (per-scan)
=====================================
Per-scan per-pixel statistics + click-to-seed signal center + auto BG placement.

Workflow:
  1. Load directory and scan range
  2. Click a point on the image to mark signal center (once)
  3. Click Compute - runs per-scan statistics
  4. For each scan, candidate background pixels are identified avoiding:
     - Stationary bright spots (persistent high intensity = powder rings,
       hot pixels, parasitic peaks)
     - The signal center and its immediate surroundings
  5. Use the scan slider to step through each scan's result
  6. Use the image slider to inspect individual raw images in that scan
  7. Save - writes ROI file with per-scan overrides

Qt window of the data tools (ctr_data_tools.py); the logic is in ctrfit.datatools.
"""
import argparse
import os
import sys

import numpy as np
from PyQt5.QtCore import Qt, QRect, QPoint, QPointF, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox,
                             QComboBox, QTextEdit, QMessageBox, QCheckBox, QProgressDialog, QSlider)

from ctrfit.app.datatools.common import run_tool
from ctrfit.datatools.bgfind import (GPU_NAME, HAS_CUPY, compute_stats, fallback_find_rois,
                                     find_bg_rois_near_point, make_exclusion_mask)
from ctrfit.datatools.images import ROI, ROIConfig, load_image, scan_directory


class BgImageView(QLabel):
    """Zoomable image view. Click to set signal center.
    Emits clicked(x, y) in image coordinates."""

    clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 500)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#1a1a1a; border:1px solid #333;")
        self.setMouseTracking(True)

        self._img_arr = None
        self._log_scale = True
        self._rois = []
        self._signal_xy = None  # (x, y) in image coords
        self._excl_mask = None
        self._show_excl = False

        self._zoom = 1.0
        self._pan = QPointF(0, 0)  # top-left offset in image-coord space
        self._img_size = (0, 0)
        self._pix_unscaled = None  # QPixmap at 1:1 pixel size

        self._drag_start = None

    def set_image(self, arr, log_scale=None):
        self._img_arr = arr
        if log_scale is not None:
            self._log_scale = log_scale
        if arr is not None:
            self._img_size = (arr.shape[1], arr.shape[0])
        self._rebuild_pixmap()
        self._render()

    def set_log_scale(self, log_scale):
        self._log_scale = log_scale
        self._rebuild_pixmap()
        self._render()

    def set_rois(self, rois):
        self._rois = rois
        self._render()

    def set_signal(self, xy):
        self._signal_xy = xy
        self._render()

    def set_exclusion_mask(self, mask, show=True):
        self._excl_mask = mask
        self._show_excl = show
        self._rebuild_pixmap()
        self._render()

    def fit_to_view(self):
        if self._img_size[0] == 0: return
        iw, ih = self._img_size
        cw, ch = self.width(), self.height()
        zx = cw / iw
        zy = ch / ih
        self._zoom = min(zx, zy)
        self._pan = QPointF(0, 0)
        self._render()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._zoom == 1.0:
            self.fit_to_view()
        else:
            self._render()

    def _rebuild_pixmap(self):
        if self._img_arr is None:
            self._pix_unscaled = None
            return
        a = self._img_arr.astype(np.float64)
        if self._log_scale:
            a = np.log10(np.maximum(a, 1.0))
        vmin = float(np.percentile(a, 1))
        vmax = float(np.percentile(a, 99.5))
        if vmax <= vmin: vmax = vmin + 1
        norm = np.clip((a - vmin) / (vmax - vmin), 0, 1)
        idx = (norm * 255).astype(np.uint8)
        h, w = idx.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = idx
        rgb[:, :, 1] = (np.sqrt(norm) * 255).astype(np.uint8)
        rgb[:, :, 2] = ((1 - norm) * 100).astype(np.uint8)

        # Overlay exclusion mask in red
        if self._show_excl and self._excl_mask is not None \
                and self._excl_mask.shape == (h, w):
            rgb[self._excl_mask, 0] = np.minimum(255, rgb[self._excl_mask, 0] + 100)
            rgb[self._excl_mask, 1] = rgb[self._excl_mask, 1] // 2
            rgb[self._excl_mask, 2] = rgb[self._excl_mask, 2] // 2

        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        self._pix_unscaled = QPixmap.fromImage(qimg.copy())

    def _render(self):
        if self._pix_unscaled is None:
            self.clear()
            return
        iw, ih = self._img_size
        sw = max(1, int(iw * self._zoom))
        sh = max(1, int(ih * self._zoom))
        scaled = self._pix_unscaled.scaled(
            sw, sh, Qt.KeepAspectRatio, Qt.FastTransformation)

        # Build final canvas matching widget size
        canvas = QPixmap(self.size())
        canvas.fill(QColor(26, 26, 26))
        painter = QPainter(canvas)

        ox = int((self.width() - sw) / 2 + self._pan.x())
        oy = int((self.height() - sh) / 2 + self._pan.y())
        painter.drawPixmap(ox, oy, scaled)

        # Overlays
        # BG ROIs - cyan
        pen = QPen(QColor(0, 200, 255)); pen.setWidth(2)
        painter.setPen(pen)
        for (x, y, w, h) in self._rois:
            painter.drawRect(QRect(
                ox + int(x * self._zoom), oy + int(y * self._zoom),
                int(w * self._zoom), int(h * self._zoom)))

        # Signal center - yellow cross
        if self._signal_xy is not None:
            sx, sy = self._signal_xy
            px = ox + int(sx * self._zoom)
            py = oy + int(sy * self._zoom)
            pen_s = QPen(QColor(255, 220, 0)); pen_s.setWidth(2)
            painter.setPen(pen_s)
            painter.drawLine(px - 10, py, px + 10, py)
            painter.drawLine(px, py - 10, px, py + 10)
            painter.drawEllipse(QPoint(px, py), 12, 12)

        painter.end()
        self.setPixmap(canvas)

    def wheelEvent(self, e):
        if self._pix_unscaled is None: return
        # Zoom at mouse position
        delta = e.angleDelta().y()
        factor = 1.15 if delta > 0 else (1 / 1.15)
        old_zoom = self._zoom
        self._zoom = max(0.1, min(50.0, self._zoom * factor))
        # Adjust pan to keep mouse point fixed
        mp = e.pos()
        iw, ih = self._img_size
        sw_old = iw * old_zoom; sh_old = ih * old_zoom
        sw_new = iw * self._zoom; sh_new = ih * self._zoom
        # Image position under mouse before zoom
        ox_old = (self.width() - sw_old) / 2 + self._pan.x()
        oy_old = (self.height() - sh_old) / 2 + self._pan.y()
        img_x = (mp.x() - ox_old) / old_zoom
        img_y = (mp.y() - oy_old) / old_zoom
        # New pan so (img_x, img_y) stays at mouse position
        ox_new = mp.x() - img_x * self._zoom
        oy_new = mp.y() - img_y * self._zoom
        self._pan = QPointF(ox_new - (self.width() - sw_new) / 2,
                            oy_new - (self.height() - sh_new) / 2)
        self._render()

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._drag_start = (e.pos(), QPointF(self._pan))
        elif e.button() == Qt.LeftButton:
            # Convert to image coords
            img_xy = self._widget_to_img(e.pos())
            if img_xy is not None:
                self.clicked.emit(*img_xy)

    def mouseMoveEvent(self, e):
        if self._drag_start is not None:
            start_pos, start_pan = self._drag_start
            dx = e.pos().x() - start_pos.x()
            dy = e.pos().y() - start_pos.y()
            self._pan = QPointF(start_pan.x() + dx, start_pan.y() + dy)
            self._render()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton:
            self._drag_start = None

    def mouseDoubleClickEvent(self, e):
        # Reset zoom
        self.fit_to_view()

    def _widget_to_img(self, pt):
        if self._img_size[0] == 0: return None
        iw, ih = self._img_size
        sw = iw * self._zoom
        sh = ih * self._zoom
        ox = (self.width() - sw) / 2 + self._pan.x()
        oy = (self.height() - sh) / 2 + self._pan.y()
        img_x = (pt.x() - ox) / self._zoom
        img_y = (pt.y() - oy) / self._zoom
        if not (0 <= img_x < iw and 0 <= img_y < ih):
            return None
        return int(img_x), int(img_y)


class BgFinderWindow(QMainWindow):
    def __init__(self, init_dir=None, scan_min=None, scan_max=None,
                 ext='h5', save_to=None):
        super().__init__()
        self.setWindowTitle("SXRD Background ROI Finder (per-scan)")
        self.resize(1400, 850)

        self._all_files = []
        self._files = []  # filtered
        self._scans_list = []  # sorted list of scan numbers in filter
        self._files_by_scan = {}  # scan_num -> [file_info, ...]

        # Per-scan results
        self._per_scan_stats = {}  # scan_num -> stats dict
        self._per_scan_boxes = {}  # scan_num -> [(x,y,w,h)]
        self._per_scan_excl = {}  # scan_num -> exclusion mask

        self._signal_xy = None  # user click, in image coords
        self._current_scan = None
        self._current_image_idx = 0  # within current scan
        self._save_to = save_to

        cw = QWidget(); self.setCentralWidget(cw)
        outer = QHBoxLayout(cw)

        # Left: controls
        ctrl = QWidget(); ctrl.setMaximumWidth(380)
        cv = QVBoxLayout(ctrl)

        # Dir
        dg = QGroupBox("Image directory")
        dl = QHBoxLayout(dg)
        self._dir = QLineEdit(); self._dir.setReadOnly(True)
        if init_dir: self._dir.setText(init_dir)
        dl.addWidget(self._dir)
        bb = QPushButton("Browse...")
        bb.clicked.connect(self._browse); dl.addWidget(bb)
        cv.addWidget(dg)

        # File type
        fg = QGroupBox("File type")
        fl = QHBoxLayout(fg)
        self._ftype = QComboBox()
        self._ftype.addItems(['h5', 'tif', 'tiff', 'All'])
        idx = self._ftype.findText(ext)
        if idx >= 0: self._ftype.setCurrentIndex(idx)
        fl.addWidget(self._ftype)
        br = QPushButton("Reload"); br.clicked.connect(self._reload_files)
        fl.addWidget(br)
        cv.addWidget(fg)

        # Scan range
        sg = QGroupBox("Scan range")
        sl = QHBoxLayout(sg)
        sl.addWidget(QLabel("From:"))
        self._sf = QSpinBox(); self._sf.setRange(0, 99999)
        if scan_min is not None: self._sf.setValue(scan_min)
        sl.addWidget(self._sf)
        sl.addWidget(QLabel("To:"))
        self._st = QSpinBox(); self._st.setRange(0, 99999)
        if scan_max is not None: self._st.setValue(scan_max)
        sl.addWidget(self._st)
        ba = QPushButton("Apply"); ba.clicked.connect(self._apply_filter)
        sl.addWidget(ba)
        cv.addWidget(sg)

        self._lbl_count = QLabel("0 images, 0 scans")
        cv.addWidget(self._lbl_count)

        # GPU status
        gpu_status = QLabel()
        if HAS_CUPY:
            gpu_status.setText(f"<b style='color:#2d8a4d'>GPU:</b> {GPU_NAME}")
        else:
            gpu_status.setText("<b style='color:#888'>GPU not available</b>")
        gpu_status.setWordWrap(True)
        cv.addWidget(gpu_status)

        # Signal center
        seg = QGroupBox("Signal center (click image to set)")
        sel = QVBoxLayout(seg)
        self._lbl_signal = QLabel("Not set - click the image")
        self._lbl_signal.setStyleSheet("color:#ffdd00;")
        sel.addWidget(self._lbl_signal)
        brow = QHBoxLayout()
        bclear = QPushButton("Clear signal")
        bclear.clicked.connect(self._clear_signal); brow.addWidget(bclear)
        sel.addLayout(brow)
        cv.addWidget(seg)

        # Compute
        cg = QGroupBox("Compute (per scan)")
        cl = QVBoxLayout(cg)
        cl.addWidget(QLabel("Lower percentile:"))
        self._pct = QSpinBox(); self._pct.setRange(1, 49); self._pct.setValue(20)
        cl.addWidget(self._pct)
        cl.addWidget(QLabel("Max images per scan:"))
        self._max_per_scan = QSpinBox(); self._max_per_scan.setRange(2, 5000); self._max_per_scan.setValue(500)
        cl.addWidget(self._max_per_scan)
        self._chk_gpu = QCheckBox("Use GPU")
        self._chk_gpu.setChecked(HAS_CUPY); self._chk_gpu.setEnabled(HAS_CUPY)
        cl.addWidget(self._chk_gpu)

        btn_compute = QPushButton("Compute all scans")
        btn_compute.setStyleSheet(
            "QPushButton{background:#2a5a8a;color:white;font-weight:bold;padding:8px;}")
        btn_compute.clicked.connect(self._compute_all)
        cl.addWidget(btn_compute)
        cv.addWidget(cg)

        # ROI options
        rg = QGroupBox("ROI placement")
        rl = QVBoxLayout(rg)
        rl.addWidget(QLabel("Box size (px):"))
        self._box_size = QSpinBox(); self._box_size.setRange(5, 500); self._box_size.setValue(30)
        rl.addWidget(self._box_size)
        rl.addWidget(QLabel("Signal exclusion radius:"))
        self._sig_radius = QSpinBox(); self._sig_radius.setRange(5, 500); self._sig_radius.setValue(40)
        rl.addWidget(self._sig_radius)
        rl.addWidget(QLabel("Search radius:"))
        self._search_r = QSpinBox(); self._search_r.setRange(20, 2000); self._search_r.setValue(200)
        rl.addWidget(self._search_r)
        rl.addWidget(QLabel("Number of BG ROIs per scan:"))
        self._n_boxes = QSpinBox(); self._n_boxes.setRange(1, 12); self._n_boxes.setValue(4)
        rl.addWidget(self._n_boxes)
        rl.addWidget(QLabel("Min clean fill fraction:"))
        self._min_fill = QDoubleSpinBox(); self._min_fill.setRange(0.1, 1.0)
        self._min_fill.setSingleStep(0.05); self._min_fill.setValue(0.7)
        rl.addWidget(self._min_fill)

        btn_find = QPushButton("Place ROIs (all scans)")
        btn_find.setStyleSheet(
            "QPushButton{background:#2d5a27;color:white;font-weight:bold;padding:8px;}")
        btn_find.clicked.connect(self._place_rois_all)
        rl.addWidget(btn_find)
        cv.addWidget(rg)

        # Display options
        dg2 = QGroupBox("Display")
        dl2 = QVBoxLayout(dg2)
        dl2.addWidget(QLabel("Show:"))
        self._view_mode = QComboBox()
        self._view_mode.addItems(['raw image', 'median', 'percentile_low',
                                   'percentile_high', 'mad', 'max'])
        self._view_mode.setCurrentText('max')
        self._view_mode.currentTextChanged.connect(self._update_display)
        dl2.addWidget(self._view_mode)
        self._chk_log = QCheckBox("Log scale")
        self._chk_log.setChecked(True)
        self._chk_log.toggled.connect(self._update_display)
        dl2.addWidget(self._chk_log)
        self._chk_show_excl = QCheckBox("Show excluded pixels (red tint)")
        self._chk_show_excl.toggled.connect(self._update_display)
        dl2.addWidget(self._chk_show_excl)
        cv.addWidget(dg2)

        # Output
        og = QGroupBox("Output")
        ol = QVBoxLayout(og)
        if save_to:
            lbl = QLabel(f"Will save to:\n{save_to}")
            lbl.setWordWrap(True); lbl.setStyleSheet("color:#aaa;")
            ol.addWidget(lbl)
            bs = QPushButton("Save and exit")
            bs.setStyleSheet(
                "QPushButton{background:#5a3a8a;color:white;font-weight:bold;padding:8px;}")
            bs.clicked.connect(self._save_and_exit); ol.addWidget(bs)
        else:
            bs = QPushButton("Save ROI file..."); bs.clicked.connect(self._save_file)
            ol.addWidget(bs)
        cv.addWidget(og)

        cv.addStretch()

        # Right: image view + sliders + log
        right = QWidget()
        rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0)

        self._img_view = BgImageView()
        self._img_view.clicked.connect(self._on_image_click)
        rv.addWidget(self._img_view, stretch=1)

        # Instructions
        self._inst = QLabel(
            "Left-click: set signal center  |  Right-drag: pan  |  "
            "Wheel: zoom  |  Double-click: fit view")
        self._inst.setStyleSheet("color:#aaa; padding:4px;")
        rv.addWidget(self._inst)

        # Scan slider
        scan_row = QHBoxLayout()
        scan_row.addWidget(QLabel("Scan:"))
        self._scan_slider = QSlider(Qt.Horizontal)
        self._scan_slider.setMinimum(0); self._scan_slider.setMaximum(0)
        self._scan_slider.valueChanged.connect(self._on_scan_slider)
        scan_row.addWidget(self._scan_slider)
        self._scan_label = QLabel("—"); self._scan_label.setMinimumWidth(120)
        scan_row.addWidget(self._scan_label)
        rv.addLayout(scan_row)

        # Image slider
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Image:"))
        self._img_slider = QSlider(Qt.Horizontal)
        self._img_slider.setMinimum(0); self._img_slider.setMaximum(0)
        self._img_slider.valueChanged.connect(self._on_img_slider)
        img_row.addWidget(self._img_slider)
        self._img_label = QLabel("—"); self._img_label.setMinimumWidth(120)
        img_row.addWidget(self._img_label)
        rv.addLayout(img_row)

        # Log
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        self._log.setStyleSheet("font-family:monospace; background:#1a1a1a; color:#ccc;")
        rv.addWidget(self._log)

        outer.addWidget(ctrl)
        outer.addWidget(right, stretch=1)

        if init_dir:
            self._reload_files()

    def _log_msg(self, msg):
        self._log.append(msg)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select image directory")
        if d:
            self._dir.setText(d); self._reload_files()

    def _reload_files(self):
        d = self._dir.text()
        if not os.path.isdir(d): return
        ext = self._ftype.currentText()
        ext_filter = None if ext == 'All' else [ext]
        self._all_files = scan_directory(d, ext_filter)
        if not self._all_files:
            self._log_msg("No files found."); return
        scans = sorted(set(f['scan'] for f in self._all_files))
        if self._sf.value() == 0: self._sf.setValue(scans[0])
        if self._st.value() == 0: self._st.setValue(scans[-1])
        self._apply_filter()
        self._log_msg(f"Loaded {len(self._all_files)} files, scans {scans[0]}-{scans[-1]}")

    def _apply_filter(self):
        s0, s1 = self._sf.value(), self._st.value()
        self._files = [f for f in self._all_files if s0 <= f['scan'] <= s1]
        self._files_by_scan = {}
        for f in self._files:
            self._files_by_scan.setdefault(f['scan'], []).append(f)
        self._scans_list = sorted(self._files_by_scan.keys())
        n = len(self._files)
        self._lbl_count.setText(f"{n} images, {len(self._scans_list)} scans")

        # Set up scan slider
        if self._scans_list:
            self._scan_slider.setMaximum(len(self._scans_list) - 1)
            self._scan_slider.setValue(0)
            self._on_scan_slider(0)

    def _on_image_click(self, x, y):
        self._signal_xy = (x, y)
        self._lbl_signal.setText(f"x={x}, y={y}")
        self._img_view.set_signal((x, y))
        self._log_msg(f"Signal center set to ({x}, {y})")

    def _clear_signal(self):
        self._signal_xy = None
        self._lbl_signal.setText("Not set - click the image")
        self._img_view.set_signal(None)

    def _compute_all(self):
        if not self._scans_list:
            QMessageBox.warning(self, "No data", "Load a directory first."); return

        self._per_scan_stats.clear()
        self._per_scan_excl.clear()
        pct = self._pct.value()
        use_gpu = self._chk_gpu.isChecked()
        max_per_scan = self._max_per_scan.value()

        total = len(self._scans_list)
        progress = QProgressDialog("Processing scans...", "Cancel", 0, total, self)
        progress.setWindowModality(Qt.WindowModal)
        self._log_msg(f"\nComputing per-scan stats for {total} scans...")

        for i, sn in enumerate(self._scans_list):
            if progress.wasCanceled(): break
            progress.setLabelText(f"Scan {sn} ({i+1}/{total})...")
            progress.setValue(i)
            QApplication.processEvents()

            files = self._files_by_scan[sn]
            # Subsample if too many
            if len(files) > max_per_scan:
                idx = np.linspace(0, len(files) - 1, max_per_scan, dtype=int)
                files = [files[j] for j in idx]

            images = []
            for fi in files:
                try:
                    images.append(load_image(fi['path'], fi['ext']))
                except Exception as e:
                    print(f"Skip {fi['path']}: {e}", file=sys.stderr)
            if not images: continue
            stats = compute_stats(images, use_gpu=use_gpu, percentile=pct)
            if stats is None: continue
            self._per_scan_stats[sn] = stats
            self._per_scan_excl[sn] = make_exclusion_mask(stats)

        progress.setValue(total); progress.close()
        backends = set(s['backend'] for s in self._per_scan_stats.values())
        self._log_msg(f"Done. {len(self._per_scan_stats)} scans computed "
                      f"(backends: {', '.join(backends)})")
        # Refresh display
        self._on_scan_slider(self._scan_slider.value())

    def _place_rois_all(self):
        if not self._per_scan_stats:
            QMessageBox.warning(self, "No stats", "Run Compute first."); return

        self._per_scan_boxes.clear()
        fallback_used = 0
        bs = self._box_size.value()
        sr = self._sig_radius.value()
        sear = self._search_r.value()
        nb = self._n_boxes.value()
        mf = self._min_fill.value()

        for sn, stats in self._per_scan_stats.items():
            excl = self._per_scan_excl.get(sn)
            if self._signal_xy is not None:
                boxes = find_bg_rois_near_point(
                    stats, self._signal_xy, excl,
                    search_radius=sear,
                    signal_exclude_radius=sr,
                    box_size=bs, n_boxes=nb, min_fill=mf)
            else:
                boxes = fallback_find_rois(stats, excl,
                    min_area=bs * bs, max_rois=nb)
                fallback_used += 1
            self._per_scan_boxes[sn] = boxes

        total_boxes = sum(len(v) for v in self._per_scan_boxes.values())
        self._log_msg(f"Placed {total_boxes} ROIs across {len(self._per_scan_boxes)} scans")
        if fallback_used:
            self._log_msg(f"  Used fallback (no signal click) for {fallback_used} scans")
        self._on_scan_slider(self._scan_slider.value())

    def _on_scan_slider(self, val):
        if not self._scans_list:
            self._scan_label.setText("—"); return
        sn = self._scans_list[val]
        self._current_scan = sn
        files = self._files_by_scan.get(sn, [])
        n_img = len(files)
        self._scan_label.setText(f"S{sn}  ({n_img} img)")

        # Reset image slider range
        self._img_slider.setMaximum(max(0, n_img - 1))
        self._img_slider.setValue(0)
        self._current_image_idx = 0
        self._update_display()

    def _on_img_slider(self, val):
        self._current_image_idx = val
        self._update_display()

    def _update_display(self):
        if self._current_scan is None: return
        sn = self._current_scan
        files = self._files_by_scan.get(sn, [])
        mode = self._view_mode.currentText()

        arr = None
        if mode == 'raw image' and files:
            idx = min(self._current_image_idx, len(files) - 1)
            info = files[idx]
            try:
                arr = load_image(info['path'], info['ext'])
            except Exception as e:
                self._log_msg(f"Load error: {e}")
            self._img_label.setText(
                f"img {idx+1}/{len(files)}  scan={info['scan']}")
        else:
            stats = self._per_scan_stats.get(sn)
            if stats is not None:
                if mode in stats:
                    arr = stats[mode]
            self._img_label.setText(f"{mode} (computed)")

        self._img_view.set_image(arr, log_scale=self._chk_log.isChecked())
        # Exclusion mask overlay
        if self._chk_show_excl.isChecked():
            self._img_view.set_exclusion_mask(
                self._per_scan_excl.get(sn), show=True)
        else:
            self._img_view.set_exclusion_mask(None, show=False)
        # ROIs
        self._img_view.set_rois(self._per_scan_boxes.get(sn, []))
        # Signal center
        self._img_view.set_signal(self._signal_xy)

    def _build_roi_config(self):
        """Build a ROIConfig with per-scan overrides."""
        cfg = ROIConfig()
        # Default: use first scan's boxes
        first_sn = self._scans_list[0] if self._scans_list else None
        if first_sn is not None and first_sn in self._per_scan_boxes:
            for i, (x, y, w, h) in enumerate(self._per_scan_boxes[first_sn]):
                cfg.default_rois.append(ROI(x, y, w, h, 'background', f'BG{i+1}'))
        # Per-scan overrides
        for sn, boxes in self._per_scan_boxes.items():
            rois = [ROI(x, y, w, h, 'background', f'BG{i+1}')
                    for i, (x, y, w, h) in enumerate(boxes)]
            for f in self._files_by_scan.get(sn, []):
                cfg.overrides[(f['scan'], f['image'])] = rois
        return cfg

    def _save_file(self):
        if not self._per_scan_boxes:
            QMessageBox.warning(self, "No ROIs", "Place ROIs first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI file", "background_rois.txt", "ROI files (*.txt)")
        if not path: return
        cfg = self._build_roi_config()
        cfg.save(path)
        self._log_msg(f"Saved to {path}")

    def _save_and_exit(self):
        if not self._per_scan_boxes:
            QMessageBox.warning(self, "No ROIs", "Place ROIs first."); return
        cfg = self._build_roi_config()
        cfg.save(self._save_to)
        self._log_msg(f"Saved, exiting.")
        QApplication.processEvents()
        self.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="Background ROI finder")
    p.add_argument('--dir', default=None)
    p.add_argument('--scan-min', type=int, default=None)
    p.add_argument('--scan-max', type=int, default=None)
    p.add_argument('--ext', default='h5')
    p.add_argument('--out', default=None)
    args = p.parse_args(argv)
    return run_tool(BgFinderWindow, init_dir=args.dir, scan_min=args.scan_min, scan_max=args.scan_max,
                    ext=args.ext, save_to=args.out)


if __name__ == '__main__':
    main()
