"""SXRD Background Subtraction GUI
=================================
For Eiger 500K area detector images (.h5 / .tif)
Designed for CTR and surface diffraction data at synchrotrons.

Usage:
    python ctr_data_tools.py   (or python -m ctrfit.app.datatools.bgsub)

Requirements:
    h5py for .h5 (Eiger) images, tifffile for .tif

Qt window of the data tools (ctr_data_tools.py); the logic is in ctrfit.datatools.
"""
import configparser
import os
import re
import tempfile

import numpy as np
from PyQt5.QtCore import QRectF, QTimer, Qt, QEvent, QEventLoop, QObject
from PyQt5.QtGui import QKeySequence, QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont
from PyQt5.QtWidgets import (QShortcut, QFormLayout, QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QGridLayout, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
                             QLineEdit, QFileDialog, QGroupBox, QCheckBox, QSplitter, QListWidget,
                             QMessageBox, QProgressDialog, QSizePolicy)

from ctrfit.app.datatools.common import (HINT_STYLE, StatusMixin, group, hint, mono_font, primary_button,
                                         row, run_tool, sidebar)
from ctrfit.datatools.images import (apply_rois, EditHistory, EditState, ImageCache, colormap_lut, display_rgb,
                                     process_image, BG_METHODS, HAS_TIFF, ROI, ROIConfig, apply_offsets,
                                     lock_offset, scan_directory, tifffile)
from ctrfit.datatools.paths import settings_file


class ImageCanvas(QLabel):
    def __init__(self, parent=None, interactive=True):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # Accept keyboard focus for space/pan

        self._img_data = None
        self._display_pixmap = None
        self._dirty = False
        self._label_font = QFont('monospace', 9)
        self.before_change = None       # called before a ROI drag or a mask stroke (undo checkpoint)
        self._rois = []
        self._interactive = interactive
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._log_scale = True
        self._vmin = None
        self._vmax = None
        self._colormap = 'gray'

        self._drag_roi_idx = -1
        self._drag_mode = None
        self._drag_start = None
        self._drag_roi_start = None

        # Zoom/pan state (in image coordinates)
        self._zoom_rect = None  # (x1, y1, x2, y2) in image coords when zoomed
        self._is_zoomed = False
        self._zoom_drag_start = None  # widget coords for zoom-box drawing
        self._zoom_drag_end = None
        self._is_drawing_zoom = False

        # Pan state
        self._is_panning = False
        self._pan_start_widget = None
        self._pan_start_offset = None
        self._space_held = False

        # Mask painting
        self._mask_paint_mode = False
        self._mask_data = None  # reference to the mask array
        self._on_mask_painted = None  # callback when mask changes
        self._mask_brush_size = 1  # pixels
        self._mask_paint_value = True  # True=mask, False=unmask, set on click

        self._on_roi_changed = None
        self._on_cursor_moved = None  # callback(x, y, value)

        # Cursor position
        self._cursor_x = -1
        self._cursor_y = -1

    def set_roi_changed_callback(self, cb):
        self._on_roi_changed = cb

    def set_cursor_callback(self, cb):
        self._on_cursor_moved = cb

    def set_mask_paint_mode(self, enabled):
        self._mask_paint_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)

    def set_mask_data(self, mask):
        """Set reference to mask array for painting."""
        self._mask_data = mask
        self.mask_changed()

    def set_mask_painted_callback(self, cb):
        self._on_mask_painted = cb

    def _paint_mask_at(self, wx, wy, determine_mode=False):
        """Paint or unpaint mask pixels near the widget position.
        If determine_mode is True, sets the paint value based on the
        center pixel (toggle: if already masked → unmask, else → mask)."""
        if self._mask_data is None or self._img_data is None:
            return
        ix, iy = self._widget_to_img(wx, wy)
        ix, iy = int(round(ix)), int(round(iy))
        h, w = self._mask_data.shape
        if not (0 <= iy < h and 0 <= ix < w):
            return

        if determine_mode:
            # Toggle: if center pixel is already masked, we unmask; else mask
            self._mask_paint_value = not self._mask_data[iy, ix]

        bs = self._mask_brush_size
        for dy in range(-bs+1, bs):
            for dx in range(-bs+1, bs):
                px, py = ix + dx, iy + dy
                if 0 <= py < h and 0 <= px < w:
                    self._mask_data[py, px] = self._mask_paint_value
        self.mask_changed()
        if self._on_mask_painted:
            self._on_mask_painted()

    def set_image(self, data, vmin=None, vmax=None):
        self._img_data = data
        if vmin is not None:
            self._vmin = vmin
        if vmax is not None:
            self._vmax = vmax
        self._update_display()

    def set_rois(self, rois):
        self._rois = rois
        self.update()

    def is_busy(self):
        """True while the mouse drags a ROI or paints the mask (callers can defer heavy work)."""
        return self._drag_roi_idx >= 0 or (self._mask_paint_mode and QApplication.mouseButtons() & Qt.LeftButton)

    def set_log_scale(self, enabled):
        self._log_scale = enabled
        self._update_display()

    def set_colormap(self, name):
        self._colormap = name
        self._update_display()

    def set_clim(self, vmin, vmax):
        self._vmin = vmin
        self._vmax = vmax
        self._update_display()

    @staticmethod
    def _build_lut(name):
        """256-entry RGB lookup table for the given colormap name."""
        return colormap_lut(name)

    def _update_display(self):
        """Mark the picture out of date; it is rebuilt once, right before the next paint, however
        many settings changed in between."""
        self._dirty = True
        self.update()

    def _ensure_rendered(self):
        if getattr(self, "_dirty", False):
            self._dirty = False
            self._render()

    def _render(self):
        if self._img_data is None:
            self._display_pixmap = None
            return
        rgb = np.ascontiguousarray(display_rgb(self._img_data, self._log_scale, self._vmin, self._vmax,
                                               self._colormap))
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._display_pixmap = QPixmap.fromImage(qimg.copy())
        self._fit_to_widget()

    def _fit_to_widget(self):
        if self._display_pixmap is None:
            return
        pw = self._display_pixmap.width()
        ph = self._display_pixmap.height()
        ww = self.width() - 4
        wh = self.height() - 4
        if pw == 0 or ph == 0:
            return

        if self._is_zoomed and self._zoom_rect is not None:
            # Zoom to the selected rectangle in image coordinates
            zx1, zy1, zx2, zy2 = self._zoom_rect
            zw = max(zx2 - zx1, 1)
            zh = max(zy2 - zy1, 1)
            self._scale = min(ww / zw, wh / zh)
            self._offset_x = -zx1 * self._scale + (ww - zw * self._scale) / 2 + 2
            self._offset_y = -zy1 * self._scale + (wh - zh * self._scale) / 2 + 2
        else:
            self._scale = min(ww / pw, wh / ph)
            self._offset_x = (ww - pw * self._scale) / 2 + 2
            self._offset_y = (wh - ph * self._scale) / 2 + 2

    def reset_zoom(self):
        """Reset to full image view."""
        self._is_zoomed = False
        self._zoom_rect = None
        self._fit_to_widget()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._ensure_rendered()
        self._fit_to_widget()
        self.update()

    def _img_to_widget(self, x, y):
        return (x * self._scale + self._offset_x,
                y * self._scale + self._offset_y)

    def _widget_to_img(self, wx, wy):
        return ((wx - self._offset_x) / self._scale,
                (wy - self._offset_y) / self._scale)

    def _mask_image(self):
        """The mask as one translucent red RGBA image (rebuilt only when the mask changed)."""
        m = self._mask_data
        if m is None:
            return None
        key = (id(m), m.shape, getattr(self, "_mask_version", 0))
        if getattr(self, "_mask_qimage_key", None) != key:
            rgba = np.zeros(m.shape + (4,), np.uint8)
            rgba[m] = (255, 40, 40, 150)
            self._mask_rgba = rgba                      # keep the buffer alive for the QImage
            self._mask_qimage = QImage(rgba.data, m.shape[1], m.shape[0], m.shape[1] * 4, QImage.Format_RGBA8888)
            self._mask_qimage_key = key
        return self._mask_qimage

    def mask_changed(self):
        """The mask array was edited in place."""
        self._mask_version = getattr(self, "_mask_version", 0) + 1
        self.update()

    def paintEvent(self, event):
        self._ensure_rendered()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#1a1a1a'))
        if self._display_pixmap is not None:
            pw = self._display_pixmap.width() * self._scale
            ph = self._display_pixmap.height() * self._scale
            painter.drawPixmap(
                int(self._offset_x), int(self._offset_y),
                int(pw), int(ph), self._display_pixmap)

        for i, roi in enumerate(self._rois):
            if roi.roi_type == 'signal':
                color = QColor(0, 255, 0, 180)
                fill = QColor(0, 255, 0, 30)
            else:
                color = QColor(255, 80, 80, 180)
                fill = QColor(255, 80, 80, 30)

            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill))
            wx, wy = self._img_to_widget(roi.x, roi.y)
            ww = roi.w * self._scale
            wh = roi.h * self._scale
            painter.drawRect(int(wx), int(wy), int(ww), int(wh))

            painter.setPen(QPen(color, 1))
            painter.setFont(self._label_font)
            label = f"{roi.name} ({roi.roi_type[0].upper()})"
            painter.drawText(int(wx + 3), int(wy - 4), label)

            if self._interactive:
                hs = 6
                painter.setBrush(QBrush(color))
                for hx, hy in [(wx, wy), (wx+ww, wy), (wx, wy+wh), (wx+ww, wy+wh)]:
                    painter.drawRect(int(hx-hs/2), int(hy-hs/2), hs, hs)

        # Draw zoom selection box
        if self._is_drawing_zoom and self._zoom_drag_start and self._zoom_drag_end:
            pen = QPen(QColor(255, 255, 0, 200), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 255, 0, 30)))
            x1, y1 = self._zoom_drag_start
            x2, y2 = self._zoom_drag_end
            painter.drawRect(int(min(x1,x2)), int(min(y1,y2)),
                             int(abs(x2-x1)), int(abs(y2-y1)))

        # Mask overlay when in mask paint mode: one scaled image, crosshairs on masked pixels when zoomed in
        if self._mask_paint_mode and self._mask_data is not None and self._display_pixmap is not None:
            h, w = self._mask_data.shape
            overlay = self._mask_image()
            painter.drawImage(QRectF(self._offset_x, self._offset_y, w * self._scale, h * self._scale), overlay)
            ix1, iy1 = self._widget_to_img(0, 0)
            ix2, iy2 = self._widget_to_img(self.width(), self.height())
            ix1 = max(0, int(ix1)); iy1 = max(0, int(iy1))
            ix2 = min(w, int(ix2) + 1); iy2 = min(h, int(iy2) + 1)
            view = self._mask_data[iy1:iy2, ix1:ix2]
            if self._scale > 4:
                painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
                half = self._scale / 2
                for py, px in zip(*np.nonzero(view)):
                    wx_p, wy_p = self._img_to_widget(px + ix1, py + iy1)
                    cx, cy = int(wx_p + half), int(wy_p + half)
                    painter.drawLine(cx - 3, cy, cx + 3, cy)
                    painter.drawLine(cx, cy - 3, cx, cy + 3)
            painter.setPen(QPen(QColor(255, 80, 80, 200), 1))
            painter.setFont(self._label_font)
            painter.drawText(10, self.height() - 8,
                             f"MASK: click=add, click masked=remove  ({int(view.sum())} masked in view)")

        painter.end()

    def _hit_test(self, wx, wy):
        hs = 10
        for i in range(len(self._rois) - 1, -1, -1):
            roi = self._rois[i]
            rx, ry = self._img_to_widget(roi.x, roi.y)
            rw = roi.w * self._scale
            rh = roi.h * self._scale
            corners = {
                'resize_tl': (rx, ry), 'resize_tr': (rx+rw, ry),
                'resize_bl': (rx, ry+rh), 'resize_br': (rx+rw, ry+rh),
            }
            for mode, (cx, cy) in corners.items():
                if abs(wx-cx) < hs and abs(wy-cy) < hs:
                    return i, mode
            if abs(wx-rx) < hs and ry < wy < ry+rh:
                return i, 'resize_l'
            if abs(wx-(rx+rw)) < hs and ry < wy < ry+rh:
                return i, 'resize_r'
            if abs(wy-ry) < hs and rx < wx < rx+rw:
                return i, 'resize_t'
            if abs(wy-(ry+rh)) < hs and rx < wx < rx+rw:
                return i, 'resize_b'
            if rx < wx < rx+rw and ry < wy < ry+rh:
                return i, 'move'
        return -1, None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_held = True
            self.setCursor(Qt.OpenHandCursor)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_held = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        # Pan mode: space held
        if self._space_held:
            self._is_panning = True
            self._pan_start_widget = (event.x(), event.y())
            self._pan_start_offset = (self._offset_x, self._offset_y)
            self.setCursor(Qt.ClosedHandCursor)
            return

        # Mask paint mode: click to toggle mask pixels
        if self._mask_paint_mode and self._mask_data is not None:
            if self.before_change:
                self.before_change()
            self._paint_mask_at(event.x(), event.y(), determine_mode=True)
            self.update()
            return

        # Interactive canvas: check ROI hit first
        if self._interactive:
            idx, mode = self._hit_test(event.x(), event.y())
            if idx >= 0:
                if self.before_change:
                    self.before_change()
                self._drag_roi_idx = idx
                self._drag_mode = mode
                self._drag_start = (event.x(), event.y())
                roi = self._rois[idx]
                self._drag_roi_start = (roi.x, roi.y, roi.w, roi.h)
                return

        # No ROI hit — start zoom box
        self._is_drawing_zoom = True
        self._zoom_drag_start = (event.x(), event.y())
        self._zoom_drag_end = (event.x(), event.y())

    def mouseMoveEvent(self, event):
        # Always track cursor position for pixel readout
        ix, iy = self._widget_to_img(event.x(), event.y())
        ix, iy = int(ix), int(iy)
        if (self._img_data is not None and
                0 <= iy < self._img_data.shape[0] and
                0 <= ix < self._img_data.shape[1]):
            self._cursor_x = ix
            self._cursor_y = iy
            val = self._img_data[iy, ix]
            if self._on_cursor_moved:
                self._on_cursor_moved(ix, iy, val)
        else:
            self._cursor_x = -1
            self._cursor_y = -1
            if self._on_cursor_moved:
                self._on_cursor_moved(-1, -1, 0)

        # Pan
        if self._is_panning and self._pan_start_widget:
            dx = event.x() - self._pan_start_widget[0]
            dy = event.y() - self._pan_start_widget[1]
            self._offset_x = self._pan_start_offset[0] + dx
            self._offset_y = self._pan_start_offset[1] + dy
            self.update()
            return

        # Mask painting (drag to paint)
        if self._mask_paint_mode and self._mask_data is not None and event.buttons() & Qt.LeftButton:
            self._paint_mask_at(event.x(), event.y())
            self.update()
            return

        # Zoom box drawing
        if self._is_drawing_zoom:
            self._zoom_drag_end = (event.x(), event.y())
            self.update()
            return

        # ROI drag (interactive only)
        if not self._interactive:
            return
        if self._drag_roi_idx >= 0 and self._drag_start is not None:
            dx = (event.x() - self._drag_start[0]) / self._scale
            dy = (event.y() - self._drag_start[1]) / self._scale
            roi = self._rois[self._drag_roi_idx]
            ox, oy, ow, oh = self._drag_roi_start
            if self._drag_mode == 'move':
                roi.x = int(ox + dx); roi.y = int(oy + dy)
            elif self._drag_mode == 'resize_br':
                roi.w = max(1, int(ow + dx)); roi.h = max(1, int(oh + dy))
            elif self._drag_mode == 'resize_tl':
                roi.x = int(ox + dx); roi.y = int(oy + dy)
                roi.w = max(1, int(ow - dx)); roi.h = max(1, int(oh - dy))
            elif self._drag_mode == 'resize_tr':
                roi.y = int(oy + dy)
                roi.w = max(1, int(ow + dx)); roi.h = max(1, int(oh - dy))
            elif self._drag_mode == 'resize_bl':
                roi.x = int(ox + dx)
                roi.w = max(1, int(ow - dx)); roi.h = max(1, int(oh + dy))
            elif self._drag_mode == 'resize_l':
                roi.x = int(ox + dx); roi.w = max(1, int(ow - dx))
            elif self._drag_mode == 'resize_r':
                roi.w = max(1, int(ow + dx))
            elif self._drag_mode == 'resize_t':
                roi.y = int(oy + dy); roi.h = max(1, int(oh - dy))
            elif self._drag_mode == 'resize_b':
                roi.h = max(1, int(oh + dy))
            self.update()
            if self._on_roi_changed:
                self._on_roi_changed()
        else:
            # Update cursor based on hover
            if self._space_held:
                self.setCursor(Qt.OpenHandCursor)
            else:
                idx, mode = self._hit_test(event.x(), event.y())
                if mode in ('resize_tl', 'resize_br'):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif mode in ('resize_tr', 'resize_bl'):
                    self.setCursor(Qt.SizeBDiagCursor)
                elif mode in ('resize_l', 'resize_r'):
                    self.setCursor(Qt.SizeHorCursor)
                elif mode in ('resize_t', 'resize_b'):
                    self.setCursor(Qt.SizeVerCursor)
                elif mode == 'move':
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, event):
        # Pan release
        if self._is_panning:
            self._is_panning = False
            self._pan_start_widget = None
            self._pan_start_offset = None
            if self._space_held:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            # Update zoom_rect to reflect the new view after panning
            if self._is_zoomed:
                # Recalculate zoom_rect from current offset/scale
                ww = self.width() - 4
                wh = self.height() - 4
                ix1, iy1 = self._widget_to_img(2, 2)
                ix2, iy2 = self._widget_to_img(ww + 2, wh + 2)
                self._zoom_rect = (ix1, iy1, ix2, iy2)
            return

        # Zoom box release
        if self._is_drawing_zoom and self._zoom_drag_start and self._zoom_drag_end:
            self._is_drawing_zoom = False
            x1w, y1w = self._zoom_drag_start
            x2w, y2w = self._zoom_drag_end
            # Only zoom if box is big enough (>10 pixels)
            if abs(x2w - x1w) > 10 and abs(y2w - y1w) > 10:
                ix1, iy1 = self._widget_to_img(min(x1w, x2w), min(y1w, y2w))
                ix2, iy2 = self._widget_to_img(max(x1w, x2w), max(y1w, y2w))
                self._zoom_rect = (ix1, iy1, ix2, iy2)
                self._is_zoomed = True
                self._fit_to_widget()
            self._zoom_drag_start = None
            self._zoom_drag_end = None
            self.update()
            return

        # ROI drag release
        if self._drag_roi_idx >= 0:
            self._drag_roi_idx = -1
            self._drag_mode = None
            self._drag_start = None
            self._drag_roi_start = None
            if self._on_roi_changed:
                self._on_roi_changed()

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset zoom."""
        if event.button() == Qt.LeftButton:
            self.reset_zoom()


class ROIListWidget(QWidget):
    def __init__(self, parent=None, on_change=None):
        super().__init__(parent)
        self._rois = []
        self._on_change = on_change
        self._current_image = None
        self._current_mask = None
        self.before_change = None       # called before an edit (undo checkpoint)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("ROIs")
        outer.addWidget(box)
        layout = QVBoxLayout(box)

        self._list = QListWidget()
        self._list.setMaximumHeight(160)
        self._list.currentRowChanged.connect(self._on_selection)
        layout.addWidget(self._list)
        btn_add, btn_rm = QPushButton("Add"), QPushButton("Remove")
        btn_add.clicked.connect(self._add_roi)
        btn_rm.clicked.connect(self._remove_roi)
        layout.addWidget(row(btn_add, btn_rm, 1, stretch_first=False))

        form = QFormLayout()
        self._name = QLineEdit()
        self._name.editingFinished.connect(self._update_from_fields)
        self._type = QComboBox()
        self._type.addItems(["signal", "background"])
        self._type.currentTextChanged.connect(self._update_from_fields)
        form.addRow("Name", self._name)
        form.addRow("Type", self._type)
        for (la, a), (lb, b) in (((("x", "_x"), ("y", "_y"))), ((("w", "_w"), ("h", "_h")))):
            pair = []
            for lab, attr in ((la, a), (lb, b)):
                sb = QSpinBox()
                sb.setRange(0 if attr in ("_x", "_y") else 1, 9999)
                sb.setPrefix(lab + " ")
                sb.setSuffix(" px")
                sb.editingFinished.connect(self._update_from_fields)
                setattr(self, attr, sb)
                pair.append(sb)
            form.addRow("Position" if a == "_x" else "Size", row(*pair, stretch_first=False))
        self._follow_combo = QComboBox()
        self._follow_combo.currentIndexChanged.connect(self._update_from_fields)
        self._follow_combo.setToolTip("Lock this ROI's position relative to another ROI.\n"
                                      "When the master moves, this ROI moves with it (rigid offset).\n"
                                      "Set to '(none)' to position this ROI independently.")
        form.addRow("Follows", self._follow_combo)
        self._btn_lock_offset = QPushButton("Lock offset")
        self._btn_lock_offset.setToolTip("Snapshot this ROI's current position relative to the master.\n"
                                         "After this, moving the master will rigidly translate this ROI.")
        self._lbl_offset = hint("offset: —")
        form.addRow("", row(self._btn_lock_offset, self._lbl_offset, stretch_first=False))
        layout.addLayout(form)

        layout.addWidget(hint("Copy to other images:"))
        grid = QGridLayout()
        grid.addWidget(QLabel("all images"), 0, 1)
        grid.addWidget(QLabel("this scan"), 0, 2)
        self._btn_apply, self._btn_apply_scan = QPushButton("All ROIs"), QPushButton("All ROIs")
        self._btn_apply_one, self._btn_apply_one_scan = QPushButton("Selected"), QPushButton("Selected")
        self._btn_apply_offset, self._btn_apply_offset_scan = QPushButton("Offset"), QPushButton("Offset")
        tips = ("Every ROI of this image", "The selected ROI",
                "Only the selected follower, placed at (its master in each image + the locked offset); "
                "masters and other followers are not touched")
        for r, (label, a, b, tip) in enumerate((("All ROIs", self._btn_apply, self._btn_apply_scan, tips[0]),
                                                ("Selected ROI", self._btn_apply_one, self._btn_apply_one_scan, tips[1]),
                                                ("Selected offset", self._btn_apply_offset, self._btn_apply_offset_scan,
                                                 tips[2])), start=1):
            grid.addWidget(QLabel(label), r, 0)
            for c, btn in ((1, a), (2, b)):
                btn.setText("Apply")
                btn.setToolTip(tip + (" → all images" if c == 1 else " → the images of the current scan"))
                grid.addWidget(btn, r, c)
        layout.addLayout(grid)
        self._chk_keep_masters = QCheckBox("Keep each image's master positions")
        self._chk_keep_masters.setToolTip(
            "Off: 'All ROIs' copies these ROIs and positions exactly.\n"
            "On: each image keeps the position of its own master ROIs (ROIs that follow no other ROI, e.g. a\n"
            "signal tracked per image); only followers are placed again from their offsets.")
        layout.addWidget(self._chk_keep_masters)

    def keep_masters(self):
        return self._chk_keep_masters.isChecked()

    def set_apply_callback(self, cb):
        self._btn_apply.clicked.connect(cb)

    def set_apply_one_callback(self, cb):
        self._btn_apply_one.clicked.connect(cb)

    def set_apply_scan_callback(self, cb):
        self._btn_apply_scan.clicked.connect(cb)

    def set_apply_one_scan_callback(self, cb):
        self._btn_apply_one_scan.clicked.connect(cb)

    def set_apply_offset_callback(self, cb):
        self._btn_apply_offset.clicked.connect(cb)

    def set_apply_offset_scan_callback(self, cb):
        self._btn_apply_offset_scan.clicked.connect(cb)

    def set_lock_offset_callback(self, cb):
        self._btn_lock_offset.clicked.connect(cb)

    def update_offset_label(self, dx, dy):
        """Update offset display (called after lock or selection change)."""
        self._lbl_offset.setText(f"offset: ({dx:+d}, {dy:+d})")

    def set_rois(self, rois):
        self._rois = rois
        self._refresh()

    def set_image(self, img, mask=None):
        """Set the current image for max pixel readout."""
        self._current_image = img
        self._current_mask = mask

    def get_rois(self):
        return self._rois

    def _refresh(self):
        cur = self._list.currentRow()
        self._list.clear()
        for r in self._rois:
            # Indicator: → if this ROI follows another
            follow = ""
            if r.follow_id is not None and 0 <= r.follow_id < len(self._rois):
                master = self._rois[r.follow_id]
                follow = f" →{master.name}"
            # Compute max pixel value in this ROI
            max_str = ""
            if self._current_image is not None:
                region = r.extract(self._current_image)
                if region.size > 0:
                    if self._current_mask is not None:
                        m = r.extract(self._current_mask)
                        if m.size == region.size:
                            valid = region[~m]
                            max_val = valid.max() if valid.size > 0 else 0
                        else:
                            max_val = region.max()
                    else:
                        max_val = region.max()
                    max_str = f" max:{max_val:.0f}"
            self._list.addItem(
                f"{r.name} [{r.roi_type[0].upper()}] ({r.x},{r.y} {r.w}x{r.h}){max_str}{follow}")
        if 0 <= cur < len(self._rois):
            self._list.setCurrentRow(cur)
        # Repopulate follow combobox so users see all current ROIs as candidates
        self._populate_follow_combo()

    def _add_roi(self):
        if self.before_change:
            self.before_change()
        n = len(self._rois) + 1
        self._rois.append(ROI(50, 50, 100, 100, 'signal', f'ROI{n}'))
        self._refresh()
        self._list.setCurrentRow(len(self._rois) - 1)
        if self._on_change:
            self._on_change()

    def _remove_roi(self):
        if self.before_change:
            self.before_change()
        idx = self._list.currentRow()
        if 0 <= idx < len(self._rois):
            self._rois.pop(idx)
            self._refresh()
            if self._on_change:
                self._on_change()

    def _populate_follow_combo(self):
        """Refresh the Follow ROI dropdown with current ROI list (excluding the
        currently-selected one). Format: "(none)", "0: ROI1", "2: ROI3", ..."""
        cur = self._list.currentRow()
        self._follow_combo.blockSignals(True)
        self._follow_combo.clear()
        self._follow_combo.addItem("(none)", userData=None)
        for i, r in enumerate(self._rois):
            if i == cur:
                continue  # can't follow itself
            self._follow_combo.addItem(f"{i}: {r.name}", userData=i)
        # Restore current selection's follow_id
        if 0 <= cur < len(self._rois):
            target_id = self._rois[cur].follow_id
            for j in range(self._follow_combo.count()):
                if self._follow_combo.itemData(j) == target_id:
                    self._follow_combo.setCurrentIndex(j)
                    break
        self._follow_combo.blockSignals(False)

    def _on_selection(self, row):
        if 0 <= row < len(self._rois):
            r = self._rois[row]
            # Block signals to prevent _update_from_fields firing mid-update
            for w in (self._name, self._type, self._x, self._y, self._w, self._h):
                w.blockSignals(True)
            self._name.setText(r.name)
            self._type.setCurrentText(r.roi_type)
            self._x.setValue(r.x)
            self._y.setValue(r.y)
            self._w.setValue(r.w)
            self._h.setValue(r.h)
            for w in (self._name, self._type, self._x, self._y, self._w, self._h):
                w.blockSignals(False)
            # Repopulate the follow combo (different ROI list now relative to selection)
            self._populate_follow_combo()
            # Show stored offset
            if r.follow_id is not None:
                self._lbl_offset.setText(f"offset: ({r.offset_dx:+d}, {r.offset_dy:+d})")
            else:
                self._lbl_offset.setText("offset: —")

    def _update_from_fields(self):
        if self.before_change:
            self.before_change()
        idx = self._list.currentRow()
        if 0 <= idx < len(self._rois):
            r = self._rois[idx]
            r.name = self._name.text()
            r.roi_type = self._type.currentText()
            r.x = self._x.value()
            r.y = self._y.value()
            r.w = self._w.value()
            r.h = self._h.value()
            # follow_id from combobox
            new_follow = self._follow_combo.currentData()
            if new_follow != r.follow_id:
                r.follow_id = new_follow
                if new_follow is None:
                    r.offset_dx = 0
                    r.offset_dy = 0
                    self._lbl_offset.setText("offset: —")
                else:
                    # Auto-lock current offset when starting to follow
                    if 0 <= new_follow < len(self._rois):
                        master = self._rois[new_follow]
                        lock_offset(r, master)
                        self._lbl_offset.setText(
                            f"offset: ({r.offset_dx:+d}, {r.offset_dy:+d})")
            self._refresh()
            self._list.setCurrentRow(idx)
            if self._on_change:
                self._on_change()

    def sync_from_canvas(self, refresh=True):
        idx = self._list.currentRow()
        if 0 <= idx < len(self._rois):
            r = self._rois[idx]
            for w in (self._x, self._y, self._w, self._h):
                w.blockSignals(True)
            self._x.setValue(r.x)
            self._y.setValue(r.y)
            self._w.setValue(r.w)
            self._h.setValue(r.h)
            for w in (self._x, self._y, self._w, self._h):
                w.blockSignals(False)
            if refresh:
                self._refresh()
                self._list.setCurrentRow(idx)


class _ScrollBlockFilter(QObject):
    """Prevents scroll wheel from changing spinbox/combobox values
    unless the widget has explicit focus (was clicked on)."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if isinstance(obj, (QSpinBox, QDoubleSpinBox, QComboBox)):
                if not obj.hasFocus():
                    event.ignore()
                    return True
        return False


class _ArrowKeyFilter(QObject):
    """Intercepts arrow keys at the application level so they
    navigate images instead of moving through widgets."""
    def __init__(self, main_window):
        super().__init__()
        self._mw = main_window

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Left, Qt.Key_Right) and self._mw.isVisible():
                # Let QLineEdit and QSpinBox keep arrows for editing
                focus = QApplication.focusWidget()
                if isinstance(focus, (QLineEdit, QDoubleSpinBox, QSpinBox)):
                    return False
                if self._mw._handle_arrow(key, event.modifiers()):
                    return True  # consumed
        return False


class MainWindow(StatusMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD Background Subtraction — Eiger 500K")
        self.setMinimumSize(1400, 800)

        self._files = []
        self._all_files = []
        self._current_idx = 0
        self._current_image = None
        self._mask = None
        self._roi_config = ROIConfig()
        self._show_full_sub = True
        self._cache = ImageCache()          # recent images + read-ahead (slow or busy disks)
        self._history = EditHistory()       # Ctrl+Z / Ctrl+Y over ROI and mask edits
        # heavy redraws (subtraction, list) at most every 40 ms while dragging or painting
        self._busy_timer = QTimer(self)
        self._busy_timer.setSingleShot(True)
        self._busy_timer.setInterval(40)
        self._busy_timer.timeout.connect(self._refresh_after_edit)
        self._busy_orig = False

        self._setup_ui()
        for seq, fn in (("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo), ("Ctrl+Shift+Z", self.redo)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self._install_key_filter()
        self._install_scroll_blocker()
        self._load_settings()

    def closeEvent(self, event):
        """Save settings on close."""
        self._save_settings()
        super().closeEvent(event)

    @staticmethod
    def _settings_path():
        """Settings file in the data tools settings folder."""
        return settings_file("sxrd_bgsub_settings.ini")

    def _save_settings(self):
        c = configparser.ConfigParser()

        c['directory'] = {
            'path': self._dir.text(),
            'file_type': self._ftype.currentText(),
            'scan_from': str(self._sf.value()),
            'scan_to': str(self._st.value()),
        }

        c['mask'] = {
            'auto_threshold': str(self._chk_auto.isChecked()),
            'threshold': str(self._thresh.value()),
            'brush_size': str(self._mask_brush.value()),
            'mask_path': getattr(self, '_last_mask_path', ''),
        }

        c['background'] = {
            'method': self._method.currentText(),
            'clip_negatives': str(self._chk_clip_neg.isChecked()),
            'save_images': str(self._chk_save_png.isChecked()),
        }

        c['display'] = {
            'log_scale': str(self._chk_log.isChecked()),
            'full_detector': str(self._chk_full.isChecked()),
            'colormap': self._cmap_combo.currentText(),
            'vmin': str(self._vmin_spin.value()),
            'vmax': str(self._vmax_spin.value()),
            'roi_visible_orig': str(self._chk_roi_orig.isChecked()),
            'roi_visible_sub': str(self._chk_roi_sub.isChecked()),
        }

        c['navigation'] = {
            'current_idx': str(self._current_idx),
        }

        c['window'] = {
            'geometry': self.saveGeometry().toBase64().data().decode(),
            'state': self.saveState().toBase64().data().decode(),
            'hsplitter': self._hsplitter.saveState().toBase64().data().decode(),
            'vsplitter': self._vsplitter.saveState().toBase64().data().decode(),
        }

        c['roi'] = {
            'roi_path': getattr(self, '_last_roi_path', ''),
        }

        try:
            with open(self._settings_path(), 'w') as f:
                c.write(f)
        except Exception:
            pass

    def _load_settings(self):
        path = self._settings_path()
        if not os.path.isfile(path):
            return

        c = configparser.ConfigParser()
        try:
            c.read(path)
        except Exception:
            return

        def get(section, key, default=''):
            try:
                return c.get(section, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return default

        def getbool(section, key, default=True):
            val = get(section, key, str(default))
            return val.lower() in ('true', '1', 'yes')

        def getfloat(section, key, default=0.0):
            try:
                return float(get(section, key, str(default)))
            except ValueError:
                return default

        def getint(section, key, default=0):
            try:
                return int(float(get(section, key, str(default))))
            except ValueError:
                return default

        # Directory and filter
        d = get('directory', 'path')
        if d:
            self._dir.setText(d)
        ft = get('directory', 'file_type', 'h5')
        idx = self._ftype.findText(ft)
        if idx >= 0:
            self._ftype.setCurrentIndex(idx)
        self._sf.setValue(getint('directory', 'scan_from', 0))
        self._st.setValue(getint('directory', 'scan_to', 999))

        # Mask
        self._chk_auto.setChecked(getbool('mask', 'auto_threshold', True))
        self._thresh.setValue(getfloat('mask', 'threshold', 1e7))
        self._mask_brush.setValue(getint('mask', 'brush_size', 1))

        # Background
        m = get('background', 'method', 'Average')
        idx = self._method.findText(m)
        if idx >= 0:
            self._method.setCurrentIndex(idx)
        self._chk_clip_neg.setChecked(getbool('background', 'clip_negatives', False))
        self._chk_save_png.setChecked(getbool('background', 'save_images', True))

        # Display
        self._chk_log.setChecked(getbool('display', 'log_scale', True))
        self._chk_full.setChecked(getbool('display', 'full_detector', True))
        cm = get('display', 'colormap', 'gray')
        idx = self._cmap_combo.findText(cm)
        if idx >= 0:
            self._cmap_combo.setCurrentIndex(idx)
        self._vmin_spin.setValue(getfloat('display', 'vmin', 0))
        self._vmax_spin.setValue(getfloat('display', 'vmax', 0))
        self._chk_roi_orig.setChecked(getbool('display', 'roi_visible_orig', True))
        self._chk_roi_sub.setChecked(getbool('display', 'roi_visible_sub', True))

        # Window geometry and splitters
        from PyQt5.QtCore import QByteArray
        geo_str = get('window', 'geometry')
        if geo_str:
            self.restoreGeometry(QByteArray.fromBase64(geo_str.encode()))
        state_str = get('window', 'state')
        if state_str:
            self.restoreState(QByteArray.fromBase64(state_str.encode()))
        hs_str = get('window', 'hsplitter')
        if hs_str:
            self._hsplitter.restoreState(QByteArray.fromBase64(hs_str.encode()))
        vs_str = get('window', 'vsplitter')
        if vs_str:
            self._vsplitter.restoreState(QByteArray.fromBase64(vs_str.encode()))

        # Reload files if directory was set
        if d and os.path.isdir(d):
            self._reload()
            saved_idx = getint('navigation', 'current_idx', 0)
            if 0 <= saved_idx < len(self._files):
                self._current_idx = saved_idx
                self._load_current()

        # Reload mask if path saved
        mask_path = get('mask', 'mask_path')
        if mask_path and os.path.isfile(mask_path):
            try:
                if mask_path.endswith('.npy'):
                    self._mask = np.load(mask_path).astype(bool)
                elif HAS_TIFF:
                    self._mask = tifffile.imread(mask_path) > 0
                self._last_mask_path = mask_path
                self._update_mask_label()
                self.say(f"Mask restored: {mask_path}")
            except Exception:
                pass

        # Reload ROI file if path saved
        roi_path = get('roi', 'roi_path')
        if roi_path and os.path.isfile(roi_path):
            try:
                self._roi_config.load(roi_path)
                self._last_roi_path = roi_path
                if self._files:
                    self._load_current()
                self.say(f"ROIs restored: {roi_path}")
            except Exception:
                pass

    def _setup_ui(self):
        # ---- left: images, ROIs, background, mask; Process at the bottom
        dg, dl = group("Images")
        self._dir = QLineEdit()
        self._dir.setPlaceholderText("folder with <prefix>_<scan>_<image>_data_<n>.h5 / .tif")
        bb = QPushButton("Browse…")
        bb.clicked.connect(self._browse)
        dl.addRow(row(self._dir, bb))
        self._ftype = QComboBox()
        self._ftype.addItems(["h5", "tif", "All"])
        self._ftype.currentTextChanged.connect(self._reload)
        dl.addRow("Type", self._ftype)
        self._sf = QSpinBox()
        self._sf.setRange(0, 999)
        self._st = QSpinBox()
        self._st.setRange(0, 999)
        self._st.setValue(999)
        bf = QPushButton("Apply")
        bf.clicked.connect(self._apply_filter)
        dl.addRow("Scans", row(self._sf, QLabel("to"), self._st, bf, stretch_first=False))
        self._lbl_count = hint("No files")
        dl.addRow(self._lbl_count)

        self._roi_w = ROIListWidget(on_change=self._on_rois_changed)
        u = self._undoable
        self._roi_w.before_change = self.checkpoint
        self._roi_w.set_apply_callback(u(self._apply_rois_all))
        self._roi_w.set_apply_one_callback(u(self._apply_one_roi_all))
        self._roi_w.set_apply_scan_callback(u(self._apply_rois_scan))
        self._roi_w.set_apply_one_scan_callback(u(self._apply_one_roi_scan))
        self._roi_w.set_lock_offset_callback(u(self._lock_selected_offset))
        self._roi_w.set_apply_offset_callback(u(self._apply_offset_all))
        self._roi_w.set_apply_offset_scan_callback(u(self._apply_offset_scan))
        ig, il = group("ROI file", form=False)
        bs, bl2 = QPushButton("Save…"), QPushButton("Load…")
        bs.clicked.connect(self._save_rois)
        bl2.clicked.connect(self._undoable(self._load_rois))
        bfind = QPushButton("Find background ROIs…")
        bfind.setToolTip("Open the background ROI finder on this folder and scan range; "
                         "its background boxes replace the background ROIs (signals are kept).")
        bfind.clicked.connect(self._undoable(self._launch_bg_finder))
        il.addWidget(row(bs, bl2, bfind, stretch_first=False))

        bg, bl = group("Background")
        self._method = QComboBox()
        self._method.addItems(list(BG_METHODS.keys()))
        self._method.currentTextChanged.connect(self._update_sub)
        self._chk_clip_neg = QCheckBox("Clip negative pixels to 0")
        self._chk_clip_neg.toggled.connect(self._update_sub)
        bl.addRow("Method", self._method)
        bl.addRow(self._chk_clip_neg)

        mg, ml = group("Dead pixel mask")
        self._chk_auto = QCheckBox("Auto, above")
        self._chk_auto.setChecked(True)
        self._thresh = QDoubleSpinBox()
        self._thresh.setRange(1e3, 1e12)
        self._thresh.setValue(1e7)
        self._thresh.setDecimals(0)
        self._thresh.setSingleStep(1e6)
        self._thresh.setSuffix(" counts")
        ml.addRow(row(self._chk_auto, self._thresh, stretch_first=False))
        self._chk_mask_paint = QCheckBox("Paint on the image")
        self._chk_mask_paint.toggled.connect(self._on_mask_paint_toggle)
        self._mask_brush = QSpinBox()
        self._mask_brush.setRange(1, 20)
        self._mask_brush.setValue(1)
        self._mask_brush.setPrefix("brush ")
        self._mask_brush.setSuffix(" px")
        self._mask_brush.setToolTip("Brush radius in pixels")
        self._mask_brush.valueChanged.connect(self._on_mask_brush_change)
        ml.addRow(row(self._chk_mask_paint, self._mask_brush, stretch_first=False))
        bm, bm_save, bm_clear = QPushButton("Load…"), QPushButton("Save…"), QPushButton("Clear")
        bm.clicked.connect(self._undoable(self._load_mask))
        bm_save.clicked.connect(self._save_mask)
        bm_clear.clicked.connect(self._undoable(self._clear_mask))
        self._lbl_mask_count = hint("0 masked pixels")
        ml.addRow(row(bm, bm_save, bm_clear, self._lbl_mask_count, stretch_first=False))

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(sidebar(dg, self._roi_w, ig, bg, mg, width=360), 1)
        self._chk_save_png = QCheckBox("Also save the subtracted images (.png)")
        self._chk_save_png.setChecked(True)
        self._chk_save_png.setToolTip("Off: only the ROI sums file is written (much faster for long scans)")
        lv.addWidget(self._chk_save_png)
        self._btn_run = primary_button("Process all images", self._run_batch, "Ctrl+R", self)
        lv.addWidget(self._btn_run)
        self._hsplitter = QSplitter(Qt.Horizontal)
        self._hsplitter.addWidget(left)

        # ---- right: navigation and display above the two images
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        bp, bn = QPushButton("◀ Prev"), QPushButton("Next ▶")
        bp.clicked.connect(lambda: self._nav(-1))
        bn.clicked.connect(lambda: self._nav(1))
        self._lbl_cur = QLabel("—")
        self._lbl_cur.setFont(mono_font())
        keys = QLabel("← →: image   Ctrl+← →: scan")
        keys.setStyleSheet(HINT_STYLE)
        right_layout.addWidget(row(bp, bn, self._lbl_cur, 1, keys, stretch_first=False))

        self._chk_log = QCheckBox("Log")
        self._chk_log.setChecked(True)
        self._chk_log.toggled.connect(self._on_log)
        self._chk_full = QCheckBox("Full detector")
        self._chk_full.setChecked(True)
        self._chk_full.toggled.connect(self._on_view_toggle)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["gray", "hot", "viridis", "inferno", "coolwarm", "jet"])
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_change)
        self._vmin_spin = QDoubleSpinBox()
        self._vmax_spin = QDoubleSpinBox()
        for w, pre in ((self._vmin_spin, "min "), (self._vmax_spin, "max ")):
            w.setRange(0, 1e12)
            w.setValue(0)
            w.setDecimals(1)
            w.setPrefix(pre)
            w.setSpecialValueText(pre + "auto")
            w.editingFinished.connect(self._on_clim_change)
            w.setMaximumWidth(150)
        self._btn_auto_clim = QPushButton("Auto")
        self._btn_auto_clim.setToolTip("Automatic colour limits")
        self._btn_auto_clim.clicked.connect(self._reset_clim)
        right_layout.addWidget(row(self._chk_log, self._chk_full, QLabel("Colormap"), self._cmap_combo,
                                   self._vmin_spin, self._vmax_spin, self._btn_auto_clim, 1, stretch_first=False))

        self._vsplitter = QSplitter(Qt.Vertical)
        oc = QWidget()
        ol = QVBoxLayout(oc)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(hint("Original (drag the ROIs)"))
        self._cv_orig = ImageCanvas(interactive=True)
        self._cv_orig.set_roi_changed_callback(self._on_canvas_drag)
        self._cv_orig.set_mask_painted_callback(self._on_mask_painted)
        self._cv_orig.set_cursor_callback(self._on_cursor_orig)
        self._cv_orig.before_change = self.checkpoint
        ol.addWidget(self._cv_orig)
        self._vsplitter.addWidget(oc)
        sc = QWidget()
        sl = QVBoxLayout(sc)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(hint("Background subtracted (preview)"))
        self._cv_sub = ImageCanvas(interactive=False)
        self._cv_sub.set_cursor_callback(self._on_cursor_sub)
        sl.addWidget(self._cv_sub)
        self._vsplitter.addWidget(sc)
        right_layout.addWidget(self._vsplitter, stretch=1)

        self._lbl_cursor = QLabel("  x: —  y: —  value: —")
        self._lbl_cursor.setFont(mono_font())
        self._chk_roi_orig = QCheckBox("ROIs on original")
        self._chk_roi_orig.setChecked(True)
        self._chk_roi_orig.toggled.connect(self._toggle_roi_orig)
        self._chk_roi_sub = QCheckBox("ROIs on subtracted")
        self._chk_roi_sub.setChecked(True)
        self._chk_roi_sub.toggled.connect(self._toggle_roi_sub)
        right_layout.addWidget(row(self._lbl_cursor, self._chk_roi_orig, self._chk_roi_sub))

        self._hsplitter.addWidget(right_container)
        self._hsplitter.setStretchFactor(0, 0)
        self._hsplitter.setStretchFactor(1, 1)
        self._hsplitter.setSizes([400, 1000])
        self.setCentralWidget(self._hsplitter)
        self.say("Pick the image folder, set signal and background ROIs, then Process all images.")

    # ---- Keys ----
    # Use an event filter to intercept arrow keys globally,
    # before child widgets (spinboxes, lists, combos) consume them.
    def _install_key_filter(self):
        self._key_filter = _ArrowKeyFilter(self)
        QApplication.instance().installEventFilter(self._key_filter)

    def _install_scroll_blocker(self):
        """Disable scroll wheel on all spinboxes and combos unless focused."""
        self._scroll_filter = _ScrollBlockFilter(self)
        for widget in self.findChildren(QSpinBox):
            widget.setFocusPolicy(Qt.StrongFocus)
            widget.installEventFilter(self._scroll_filter)
        for widget in self.findChildren(QDoubleSpinBox):
            widget.setFocusPolicy(Qt.StrongFocus)
            widget.installEventFilter(self._scroll_filter)
        for widget in self.findChildren(QComboBox):
            widget.setFocusPolicy(Qt.StrongFocus)
            widget.installEventFilter(self._scroll_filter)

    def _handle_arrow(self, key, modifiers):
        """Called by the event filter when an arrow key is pressed."""
        # Don't intercept if a QLineEdit or QSpinBox has focus and user is editing
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QDoubleSpinBox)):
            return False
        # Allow spinbox arrow keys only when the spinbox itself is focused
        if isinstance(focus, QSpinBox):
            return False

        if key == Qt.Key_Left:
            if modifiers & Qt.ControlModifier:
                self._nav_scan(-1)
            else:
                self._nav(-1)
            return True
        elif key == Qt.Key_Right:
            if modifiers & Qt.ControlModifier:
                self._nav_scan(1)
            else:
                self._nav(1)
            return True
        return False

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            self._handle_arrow(event.key(), event.modifiers())
        else:
            super().keyPressEvent(event)

    # ---- Files ----
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select data directory")
        if d:
            self._dir.setText(d)
            self._reload()

    def _reload(self):
        d = self._dir.text()
        if not os.path.isdir(d):
            return
        ft = self._ftype.currentText()
        ext = None if ft == "All" else [ft]
        self._all_files = scan_directory(d, ext)
        self._apply_filter()

    def _apply_filter(self):
        s0, s1 = self._sf.value(), self._st.value()
        self._files = [f for f in self._all_files if s0 <= f['scan'] <= s1]
        n = len(self._files)
        scans = len(set(f['scan'] for f in self._files))
        self._lbl_count.setText(f"{n} images, {scans} scans")
        if n > 0:
            self._current_idx = 0
            self._load_current()

    # ---- Navigation ----
    def _nav(self, step):
        if not self._files:
            return
        self._save_cur_rois()
        self._current_idx = max(0, min(len(self._files) - 1, self._current_idx + step))
        self._load_current()

    def _nav_scan(self, direction):
        if not self._files:
            return
        self._save_cur_rois()
        cur_info = self._files[self._current_idx]
        cur_scan = cur_info['scan']
        cur_image = cur_info['image']

        # Find the target scan number
        target_scan = None
        if direction > 0:
            for i in range(self._current_idx + 1, len(self._files)):
                if self._files[i]['scan'] > cur_scan:
                    target_scan = self._files[i]['scan']
                    break
        else:
            for i in range(self._current_idx - 1, -1, -1):
                if self._files[i]['scan'] < cur_scan:
                    target_scan = self._files[i]['scan']
                    break

        if target_scan is None:
            return

        # Find all file indices belonging to target scan
        target_indices = [i for i, f in enumerate(self._files)
                          if f['scan'] == target_scan]
        if not target_indices:
            return

        # Pick the one with the closest image number (prefer exact match)
        best = min(target_indices,
                   key=lambda i: abs(self._files[i]['image'] - cur_image))
        self._current_idx = best
        self._load_current()

    def _load_current(self):
        if not self._files:
            return
        info = self._files[self._current_idx]
        self._lbl_cur.setText(
            f"Scan {info['scan']:03d} / Img {info['image']:03d}  "
            f"[{self._current_idx+1}/{len(self._files)}]")
        try:
            self._current_image = self._cache.get(info['path'], info['ext'])
            i = self._current_idx
            self._cache.prefetch([self._files[k] for k in (i + 1, i + 2, i - 1) if 0 <= k < len(self._files)])
        except Exception as e:
            self.say(f"Error: {e}")
            self._current_image = None
            return

        if self._chk_auto.isChecked() and self._current_image is not None:
            self._mask = self._current_image > self._thresh.value()

        disp = np.array(self._current_image, dtype=float, copy=True)
        if self._mask is not None and self._mask.shape == disp.shape:
            disp[self._mask] = 0

        self._cv_orig.set_log_scale(self._chk_log.isChecked())
        self._cv_orig.set_colormap(self._cmap_combo.currentText())
        vmin = self._vmin_spin.value() if self._vmin_spin.value() > 0 else None
        vmax = self._vmax_spin.value() if self._vmax_spin.value() > 0 else None
        self._cv_orig.set_clim(vmin, vmax)
        self._cv_orig.set_image(disp)

        rois = self._roi_config.get_rois(info['scan'], info['image'])
        # Only auto-apply offsets when this image has no override yet (fresh
        # defaults). If an override exists, the user has already positioned
        # the ROIs for this frame - don't overwrite their manual edits.
        if not self._roi_config.has_override(info['scan'], info['image']):
            apply_offsets(rois)
        self._roi_w.set_image(self._current_image, self._mask)
        self._roi_w.set_rois(rois)
        self._cv_orig.set_rois(rois if self._chk_roi_orig.isChecked() else [])
        self._update_sub()
        self.say(f"Loaded: {os.path.basename(info['path'])}")

    def _save_cur_rois(self):
        if not self._files:
            return
        info = self._files[self._current_idx]
        self._roi_config.set_override(info['scan'], info['image'], self._roi_w.get_rois())

    # ---- ROI callbacks ----
    def _on_rois_changed(self):
        rois = self._roi_w.get_rois()
        # If the user is editing a MASTER (non-follower), propagate to followers.
        # If the user is editing a follower (e.g. dodging a powder ring), leave
        # everything else alone so the manual edit sticks.
        sel_idx = self._roi_w._list.currentRow()
        if 0 <= sel_idx < len(rois) and rois[sel_idx].follow_id is None:
            apply_offsets(rois)
        self._cv_orig.set_rois(rois if self._chk_roi_orig.isChecked() else [])
        self._update_sub()

    def _on_canvas_drag(self):
        busy = self._cv_orig.is_busy()
        self._roi_w.sync_from_canvas(refresh=not busy)
        rois = self._roi_w.get_rois()
        sel_idx = self._roi_w._list.currentRow()
        if 0 <= sel_idx < len(rois) and rois[sel_idx].follow_id is None:
            apply_offsets(rois)
            if not busy:
                self._roi_w.set_rois(rois)  # push back so widget display refreshes
        self._cv_orig.set_rois(rois if self._chk_roi_orig.isChecked() else [])
        if busy:
            self._defer_refresh()
        else:
            self._busy_timer.stop()
            self._update_sub()

    def _defer_refresh(self, orig=False):
        """Redo the subtraction (and the ROI list, the original view if orig) soon, at most every 40 ms."""
        self._busy_orig = self._busy_orig or orig
        if not self._busy_timer.isActive():
            self._busy_timer.start()

    def _refresh_after_edit(self):
        if self._busy_orig and self._current_image is not None:
            self._show_original()
        self._busy_orig = False
        if not self._cv_orig.is_busy():
            self._roi_w._refresh()
        self._update_sub()

    def _show_original(self):
        disp = np.array(self._current_image, dtype=float, copy=True)
        if self._mask is not None and self._mask.shape == disp.shape:
            disp[self._mask] = 0
        self._cv_orig.set_image(disp)

    # ---- Undo / redo ----
    def _state(self):
        return EditState(self._roi_config, self._roi_w.get_rois(), self._mask)

    def checkpoint(self):
        """Remember the ROIs and the mask before a change (Ctrl+Z goes back here)."""
        self._history.checkpoint(self._state())

    def _undoable(self, fn):
        def run(*_):
            self.checkpoint()
            return fn()
        return run

    def undo(self):
        self._restore(self._history.undo(self._state()), "Undone")

    def redo(self):
        self._restore(self._history.redo(self._state()), "Redone")

    def _restore(self, state, what):
        if state is None:
            self.say(f"Nothing to {'undo' if what == 'Undone' else 'redo'}")
            return
        rois, mask = state.apply_to(self._roi_config)
        self._mask = mask
        self._cv_orig.set_mask_data(self._mask)
        self._update_mask_label()
        self._roi_w.set_rois(rois)
        self._cv_orig.set_rois(rois if self._chk_roi_orig.isChecked() else [])
        if self._current_image is not None:
            self._show_original()
        self._update_sub()
        self.say(f"{what} (Ctrl+Z undo, Ctrl+Y redo)")

    def _apply_rois_all(self):
        """Copy the ROIs of this image to every loaded image (and make them the defaults). With 'Keep
        each image's master positions', images keep their own master (e.g. tracked signal) positions
        and only the followers move."""
        keep = self._roi_w.keep_masters()
        reply = QMessageBox.question(
            self, "Apply all ROIs to all images?",
            f"Replace the ROIs of all {len(self._files)} loaded images with the ROIs of this image?\n\n"
            + ("Each image keeps the position of its own master ROIs (those that follow no other ROI); "
               "followers are placed from their offsets." if keep else
               "Every image gets exactly these ROIs and positions; edits made on other images are replaced.")
            + "\n\nCtrl+Z undoes this.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply != QMessageBox.Yes:
            self.say("Apply all cancelled")
            return
        keys = [(f['scan'], f['image']) for f in self._files]
        n = apply_rois(self._roi_config, keys, self._roi_w.get_rois(), keep_masters=keep, set_default=True)
        self.say(f"ROIs applied to all {n} images" + (" (each image's master positions kept)" if keep else ""))

    def _apply_one_roi_all(self):
        """Update only the selected ROI across all defaults and overrides."""
        rois = self._roi_w.get_rois()
        idx = self._roi_w._list.currentRow()
        if not (0 <= idx < len(rois)):
            self.say("No ROI selected")
            return
        updated_roi = rois[idx]

        # Update in defaults (by index position)
        if idx < len(self._roi_config.default_rois):
            self._roi_config.default_rois[idx] = updated_roi.copy()

        # Update in all overrides that have this index
        for key, override_rois in self._roi_config.overrides.items():
            if idx < len(override_rois):
                override_rois[idx] = updated_roi.copy()

        self.say(f"'{updated_roi.name}' (ROI {idx+1}) applied to all images")

    def _apply_rois_scan(self):
        """Copy the ROIs of this image to every image of the current scan (see _apply_rois_all)."""
        if not self._files:
            self.say("No files loaded")
            return
        keep = self._roi_w.keep_masters()
        cur_scan = self._files[self._current_idx]['scan']
        keys = [(f['scan'], f['image']) for f in self._files if f['scan'] == cur_scan]
        n = apply_rois(self._roi_config, keys, self._roi_w.get_rois(), keep_masters=keep)
        self.say(f"ROIs applied to the {n} images of scan {cur_scan}"
                 + (" (each image's master positions kept)" if keep else ""))

    def _apply_one_roi_scan(self):
        """Update only the selected ROI in every image of the CURRENT scan."""
        if not self._files:
            self.say("No files loaded")
            return
        rois = self._roi_w.get_rois()
        idx = self._roi_w._list.currentRow()
        if not (0 <= idx < len(rois)):
            self.say("No ROI selected")
            return
        updated_roi = rois[idx]
        cur_scan = self._files[self._current_idx]['scan']
        count = 0
        for f in self._files:
            if f['scan'] != cur_scan:
                continue
            key = (f['scan'], f['image'])
            # Create override from current defaults if this image has none yet
            if key not in self._roi_config.overrides:
                self._roi_config.overrides[key] = [
                    r.copy() for r in self._roi_config.default_rois]
            if idx < len(self._roi_config.overrides[key]):
                self._roi_config.overrides[key][idx] = updated_roi.copy()
                count += 1
        self.say(f"'{updated_roi.name}' applied to scan {cur_scan} ({count} images)")

    def _apply_offset_all(self):
        """Reposition the currently-selected follower ROI in every existing
        override, placing it at (master_position + offset) using each frame's
        own master position. Does not touch other ROIs.

        Master is identified by NAME (not list index) so this works even if
        the ROI list order has changed across overrides.
        """
        self._apply_offset_to_files(self._files, scope_label="frames")

    def _apply_offset_scan(self):
        """Same as _apply_offset_all but limited to the current scan."""
        if not self._files:
            self.say("No files loaded")
            return
        cur_scan = self._files[self._current_idx]['scan']
        scan_files = [f for f in self._files if f['scan'] == cur_scan]
        self._apply_offset_to_files(
            scan_files, scope_label=f"frames in scan {cur_scan}")

    def _apply_offset_to_files(self, files, scope_label=""):
        widget_rois = self._roi_w.get_rois()
        idx = self._roi_w._list.currentRow()
        if not (0 <= idx < len(widget_rois)):
            self.say("No ROI selected")
            return
        follower = widget_rois[idx]
        if follower.follow_id is None:
            self.say(f"'{follower.name}' has no master to follow - "
                f"set 'Follow ROI' first")
            return
        if not (0 <= follower.follow_id < len(widget_rois)):
            self.say(f"'{follower.name}': invalid follow target")
            return
        master_name = widget_rois[follower.follow_id].name
        widget_master_pos = widget_rois[follower.follow_id].center()

        applied = 0
        no_master = 0
        created = 0
        for f in files:
            key = (f['scan'], f['image'])
            override = self._roi_config.overrides.get(key)
            if override is None:
                # No override yet - create one from defaults so this follower
                # has a place to live. Master comes from defaults if present;
                # otherwise from widget.
                new_override = [r.copy() for r in self._roi_config.default_rois]
                follower_idx = self._find_roi_index_by_name(new_override, follower.name)
                if follower_idx is None:
                    new_override.append(follower.copy())
                    follower_idx = len(new_override) - 1
                else:
                    new_override[follower_idx] = follower.copy()
                master_idx = self._find_roi_index_by_name(new_override, master_name)
                if master_idx is None:
                    new_override[follower_idx].set_center(*widget_master_pos)
                    no_master += 1
                else:
                    new_override[follower_idx].follow_id = master_idx
                    mcx, mcy = new_override[master_idx].center()
                    new_override[follower_idx].set_center(
                        mcx + follower.offset_dx,
                        mcy + follower.offset_dy)
                self._roi_config.overrides[key] = new_override
                applied += 1
                created += 1
            else:
                master_idx = self._find_roi_index_by_name(override, master_name)
                if master_idx is None:
                    no_master += 1
                    continue
                follower_idx = self._find_roi_index_by_name(override, follower.name)
                if follower_idx is None:
                    new_follower = follower.copy()
                    new_follower.follow_id = master_idx
                    override.append(new_follower)
                    follower_idx = len(override) - 1
                else:
                    override[follower_idx].follow_id = master_idx
                    override[follower_idx].offset_dx = follower.offset_dx
                    override[follower_idx].offset_dy = follower.offset_dy
                mcx, mcy = override[master_idx].center()
                override[follower_idx].set_center(
                    mcx + follower.offset_dx,
                    mcy + follower.offset_dy)
                applied += 1

        msg = (f"'{follower.name}' applied to {applied} {scope_label} "
               f"(master={master_name}, offset=("
               f"{follower.offset_dx:+d},{follower.offset_dy:+d}))")
        if created:
            msg += f"; created {created} new overrides"
        if no_master:
            msg += f"; {no_master} frames missing master '{master_name}'"
        self.say(msg)
        # Refresh display so user sees the result on the current frame
        self._load_current()

    @staticmethod
    def _find_roi_index_by_name(rois, name):
        for i, r in enumerate(rois):
            if r.name == name:
                return i
        return None

    def _lock_selected_offset(self):
        """Capture the currently-selected ROI's offset relative to its master."""
        rois = self._roi_w.get_rois()
        idx = self._roi_w._list.currentRow()
        if not (0 <= idx < len(rois)):
            return
        r = rois[idx]
        if r.follow_id is None:
            self.say(f"{r.name} has no master to follow - set 'Follow ROI' first")
            return
        if not (0 <= r.follow_id < len(rois)) or r.follow_id == idx:
            self.say(f"{r.name}: invalid follow target")
            return
        master = rois[r.follow_id]
        lock_offset(r, master)
        self._roi_w.update_offset_label(r.offset_dx, r.offset_dy)
        self.say(f"Locked {r.name} → {master.name} at offset "
            f"({r.offset_dx:+d}, {r.offset_dy:+d})")
        if hasattr(self._roi_w, '_on_change') and self._roi_w._on_change:
            self._roi_w._on_change()

    def _on_log(self, checked):
        self._cv_orig.set_log_scale(checked)
        self._cv_sub.set_log_scale(checked)
        self._update_sub()

    def _on_view_toggle(self, checked):
        self._show_full_sub = checked
        self._update_sub()

    def _on_cmap_change(self, name):
        self._cv_orig.set_colormap(name)
        self._cv_sub.set_colormap(name)

    def _on_clim_change(self):
        vmin = self._vmin_spin.value() if self._vmin_spin.value() > 0 else None
        vmax = self._vmax_spin.value() if self._vmax_spin.value() > 0 else None
        self._cv_orig.set_clim(vmin, vmax)
        self._cv_sub.set_clim(vmin, vmax)

    def _reset_clim(self):
        self._vmin_spin.setValue(0)
        self._vmax_spin.setValue(0)
        self._cv_orig.set_clim(None, None)
        self._cv_sub.set_clim(None, None)
        self._cv_orig._update_display()
        self._cv_sub._update_display()

    def _on_cursor_orig(self, x, y, val):
        if x < 0:
            self._lbl_cursor.setText("  x: —  y: —  value: —")
        else:
            self._lbl_cursor.setText(f"  x: {x}  y: {y}  value: {val:.4g}")

    def _on_cursor_sub(self, x, y, val):
        if x < 0:
            self._lbl_cursor.setText("  x: —  y: —  value: —")
        else:
            self._lbl_cursor.setText(f"  x: {x}  y: {y}  value: {val:.4g}  (subtracted)")

    def _toggle_roi_orig(self, checked):
        if checked:
            rois = self._roi_w.get_rois()
            self._cv_orig.set_rois(rois)
        else:
            self._cv_orig.set_rois([])

    def _toggle_roi_sub(self, checked):
        if checked:
            rois = self._roi_w.get_rois()
            self._cv_sub.set_rois(rois)
        else:
            self._cv_sub.set_rois([])

    def _set_rois_respecting_toggles(self, rois):
        """Set ROIs on canvases respecting visibility toggles."""
        self._cv_orig.set_rois(rois if self._chk_roi_orig.isChecked() else [])
        self._cv_sub.set_rois(rois if self._chk_roi_sub.isChecked() else [])

    # ---- Subtraction ----
    def _update_sub(self):
        if self._current_image is None:
            return
        rois = self._roi_w.get_rois()
        sig = [r for r in rois if r.roi_type == 'signal']
        bg = [r for r in rois if r.roi_type == 'background']

        if not sig or not bg:
            disp = np.array(self._current_image, dtype=float, copy=True)
            if self._mask is not None and self._mask.shape == disp.shape:
                disp[self._mask] = 0
            self._cv_sub.set_log_scale(self._chk_log.isChecked())
            self._cv_sub.set_image(disp)
            self._cv_sub.set_rois(rois if self._chk_roi_sub.isChecked() else [])
            return

        method = BG_METHODS[self._method.currentText()]
        sub = method(self._current_image, sig, bg, self._mask)

        if self._chk_clip_neg.isChecked():
            sub = np.clip(sub, 0, None)

        if self._show_full_sub:
            disp = sub
        else:
            disp = np.zeros_like(sub)
            for roi in sig:
                y1 = max(0, roi.y); x1 = max(0, roi.x)
                y2 = min(disp.shape[0], roi.y+roi.h)
                x2 = min(disp.shape[1], roi.x+roi.w)
                disp[y1:y2, x1:x2] = sub[y1:y2, x1:x2]

        self._cv_sub.set_log_scale(self._chk_log.isChecked())
        self._cv_sub.set_image(disp)
        self._cv_sub.set_rois(rois if self._chk_roi_sub.isChecked() else [])

    # ---- Mask ----
    def _load_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load mask", "", "NumPy (*.npy);;TIFF (*.tif);;All (*)")
        if not path:
            return
        try:
            if path.endswith('.npy'):
                self._mask = np.load(path).astype(bool)
            else:
                self._mask = tifffile.imread(path) > 0
            self._last_mask_path = path
            self._update_mask_label()
            self._cv_orig.set_mask_data(self._mask)
            self.say(f"Mask loaded: {self._mask.sum()} dead pixels")
        except Exception as e:
            self.say(str(e), error=True)

    def _save_mask(self):
        if self._mask is None:
            self.say("No mask to save.", error=True)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save mask", "eiger_mask.npy",
                                               "NumPy (*.npy);;TIFF (*.tif)")
        if not path:
            return
        try:
            if path.endswith('.npy'):
                np.save(path, self._mask)
            elif path.endswith(('.tif', '.tiff')):
                if HAS_TIFF:
                    tifffile.imwrite(path, self._mask.astype(np.uint8))
                else:
                    np.save(path.replace('.tif', '.npy'), self._mask)
            self.say(f"Mask saved: {path}")
        except Exception as e:
            self.say(str(e), error=True)

    def _clear_mask(self):
        if self._mask is not None:
            self._mask[:] = False
            self._update_mask_label()
            self._cv_orig.set_mask_data(self._mask)
            self._load_current()
            self.say("Mask cleared")

    def _on_mask_paint_toggle(self, enabled):
        self._cv_orig.set_mask_paint_mode(enabled)
        if enabled:
            # Ensure mask array exists
            if self._current_image is not None:
                if self._mask is None or self._mask.shape != self._current_image.shape:
                    self._mask = np.zeros(self._current_image.shape, dtype=bool)
                    if self._chk_auto.isChecked():
                        self._mask |= self._current_image > self._thresh.value()
                self._cv_orig.set_mask_data(self._mask)
            self.say("Mask paint ON — click/drag on image to mark dead pixels")
        else:
            self.say("Mask paint OFF")

    def _on_mask_brush_change(self, val):
        self._cv_orig._mask_brush_size = val

    def _on_mask_painted(self):
        """Called by canvas when mask pixels are painted."""
        self._update_mask_label()
        self._defer_refresh(orig=True)          # at most every 40 ms while the brush moves

    def _update_mask_label(self):
        n = int(self._mask.sum()) if self._mask is not None else 0
        self._lbl_mask_count.setText(f"{n} masked pixels")

    # ---- ROI I/O ----
    def _save_rois(self):
        self._save_cur_rois()
        default_name = self._last_roi_path if hasattr(self, '_last_roi_path') else "ROIs_default.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Save ROIs", default_name, "Text (*.txt)")
        if path:
            self._roi_config.save(path)
            self._last_roi_path = path
            self.say(f"Saved: {path}")

    def _load_rois(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load ROIs", "", "Text (*.txt);;All (*)")
        if path:
            try:
                self._roi_config.load(path)
                self._last_roi_path = path
                self._load_current()
                self.say(f"Loaded: {path}")
            except Exception as e:
                self.say(str(e), error=True)

    def _launch_bg_finder(self):
        """Open the background ROI finder (modal) on this directory and scan range. When it is
        closed, load the ROI file it saved and merge it into the current config as defaults
        (replacing background ROIs only, keeping signals)."""
        d = self._dir.text() if hasattr(self, '_dir') else ''
        if not d or not os.path.isdir(d):
            self.say("Load an image directory in the main window first.", error=True)
            return
        s0 = self._sf.value() if hasattr(self, '_sf') else 0
        s1 = self._st.value() if hasattr(self, '_st') else 99999
        ext = self._ftype.currentText() if hasattr(self, '_ftype') else 'h5'
        if ext == 'All':
            ext = 'h5'
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='_bg_rois.txt', delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            os.unlink(tmp_path)  # the finder creates it when it saves
        except OSError:
            pass
        from .bg_finder import BgFinderWindow
        self.say("Opening background ROI finder...")
        finder = BgFinderWindow(init_dir=d, scan_min=s0, scan_max=s1, ext=ext, save_to=tmp_path)
        finder.setAttribute(Qt.WA_DeleteOnClose)
        finder.setWindowModality(Qt.ApplicationModal)
        loop = QEventLoop()
        finder.destroyed.connect(loop.quit)
        finder.show()
        loop.exec_()

        # Check if a file was written
        if not os.path.exists(tmp_path):
            self.say("Background finder closed without saving.")
            return

        # Load proposed ROIs
        proposed = ROIConfig()
        try:
            proposed.load(tmp_path)
        except Exception as e:
            self.say(f"Could not parse output: {e}", error=True)
            return
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass

        if not proposed.default_rois:
            self.say("No background ROIs were proposed.")
            return

        # Merge: keep existing signal ROIs from defaults, replace background ROIs
        # with the proposed ones
        signal_rois = [r for r in self._roi_config.default_rois
                       if r.roi_type == 'signal']
        bg_rois = [r for r in proposed.default_rois
                   if r.roi_type == 'background']

        # Apply to current scan range as overrides too, so the user sees them
        # right away in the main window
        new_defaults = signal_rois + bg_rois
        self._roi_config.default_rois = [r.copy() for r in new_defaults]

        # Apply as overrides to all images in the current scan range
        # so the new BG ROIs are visible immediately
        for f in self._files:
            self._roi_config.overrides[(f['scan'], f['image'])] = [
                r.copy() for r in new_defaults]

        self._load_current()
        self.say(f"Loaded {len(bg_rois)} background ROIs from finder, "
            f"applied to {len(self._files)} images")

    # ---- Batch ----
    def _run_batch(self):
        if not self._files:
            self.say("No files loaded.", error=True)
            return
        self._save_cur_rois()

        # Ask user for output text file name
        default_name = "roi_sums.txt"
        if hasattr(self, '_last_roi_path') and self._last_roi_path:
            default_name = os.path.splitext(os.path.basename(self._last_roi_path))[0] + "_sums.txt"
        sum_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI sums as...", default_name, "Text files (*.txt);;All (*)")
        if not sum_path:
            return

        out_dir = os.path.dirname(sum_path)
        if not out_dir:
            out_dir = '.'
        txt_stem = os.path.splitext(os.path.basename(sum_path))[0]

        save_png = self._chk_save_png.isChecked()
        img_subdir = os.path.join(out_dir, txt_stem + "_images")      # named after the sums file
        if save_png:
            os.makedirs(img_subdir, exist_ok=True)

        method = BG_METHODS[self._method.currentText()]
        n = len(self._files)

        progress = QProgressDialog("Processing...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        # Collect all ROI names
        all_names = set()
        for info in self._files:
            for r in self._roi_config.get_rois(info['scan'], info['image']):
                all_names.add(r.name)
        roi_names = sorted(all_names)
        err_names = sorted({f"{r.name}_err" for info in self._files
                            for r in self._roi_config.get_rois(info['scan'], info['image']) if r.roi_type == 'signal'})

        sum_rows = []
        for i, info in enumerate(self._files):
            if progress.wasCanceled():
                break
            progress.setValue(i)

            self._cache.prefetch(self._files[i + 1:i + 3])            # read the next images meanwhile
            try:
                img = self._cache.get(info['path'], info['ext'])
            except Exception:
                continue

            mask = None
            if self._chk_auto.isChecked():
                mask = img > self._thresh.value()
            if self._mask is not None and self._mask.shape == img.shape:   # with the loaded / painted mask
                mask = self._mask.copy() if mask is None else (mask | self._mask)

            rois = self._roi_config.get_rois(info['scan'], info['image'])
            # Saved overrides are the user's positions for that image; offsets only for fresh defaults.
            if not self._roi_config.has_override(info['scan'], info['image']):
                apply_offsets(rois)
            sub, sums = process_image(img, rois, method, mask, self._chk_clip_neg.isChecked())

            if save_png:
                self._save_png(sub, info, img_subdir)

            row = ([info['scan'], info['image']] + [sums.get(n, 0.0) for n in roi_names]
                   + [sums.get(n, float('nan')) for n in err_names])
            sum_rows.append(row)

        progress.setValue(n)

        # Save sums to the user-chosen path (tab-separated for Excel)
        with open(sum_path, 'w') as f:
            header = ["scan", "image"] + roi_names + err_names       # *_err: counting error of a signal ROI
            f.write("\t".join(header) + "\n")
            for row in sum_rows:
                parts = [f"{row[0]:03d}", f"{row[1]:03d}"]
                parts += [f"{v:.6g}" for v in row[2:]]
                f.write("\t".join(parts) + "\n")

        self.say(f"Processed {len(sum_rows)} images. Sums: {sum_path}" + (f"; images in {img_subdir}" if save_png else ""))

    def _save_png(self, sub, info, img_subdir):
        """The subtracted image as a colour-mapped PNG, as shown in the preview (else a TIFF)."""
        base = os.path.basename(info['path'])
        out_name = re.sub(r'\.(h5|tif|tiff)$', '_sub.png', base)
        vmin = self._vmin_spin.value() or None
        vmax = self._vmax_spin.value() or None
        try:
            rgb = display_rgb(sub, self._chk_log.isChecked(), vmin, vmax, self._cmap_combo.currentText(), exact=True)
            try:
                from PIL import Image
                Image.fromarray(rgb, 'RGB').save(os.path.join(img_subdir, out_name), compress_level=1)
            except ImportError:
                if HAS_TIFF:
                    tifffile.imwrite(os.path.join(img_subdir, out_name.replace('.png', '.tif')), rgb, compression='zlib')
        except Exception:
            if HAS_TIFF:
                tifffile.imwrite(os.path.join(img_subdir, re.sub(r'\.png$', '.tif', out_name)), sub.astype(np.float32))


def main(argv=None):
    return run_tool(MainWindow)


if __name__ == '__main__':
    main()
