# ui_qt_ocr_with_lstm.py — PySide6 UI integrated with Keras LSTM model
#
# Features:
# - Loads a Keras (.keras / SavedModel) seq2seq or classification model
# - Tries to auto-detect token mappings (token_to_idx / idx_to_token) from JSON files or user selection
# - Removes fuzzy/phonetic logic — predictions come only from the model
# - Provides a simple UI to enter a medicine string and get the model's reconstructed output
# - Shows a confidence meter derived from model probabilities (if available)
#
# Usage:
# - Place your model at DEFAULT_MODEL_PATH or load with File -> Load model
# - Optionally place token_to_idx.json and idx_to_token.json in the same folder as the model
# - If mappings are not found, the script will attempt a best-effort char-level mapping (may not match training)

import os
import sys
import json
from typing import Tuple, Optional, Dict, Any

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QProgressBar, QTableWidget, QTableWidgetItem,
    QMessageBox, QScrollArea, QCheckBox, QStatusBar, QMenuBar, QFileDialog, QFrame
)

# Adjust these defaults if your files are in different locations
DEFAULT_MODEL_PATH = "/mnt/data/Word_Reconstruct_LSTM (4).keras"
DEFAULT_TOKEN_TO_IDX = None  # try auto-detect next to model
DEFAULT_IDX_TO_TOKEN = None
DEFAULT_MAX_LEN = None
MIN_CONFIDENCE = 65.0

_tf = None

def lazy_load_tf():
    global _tf
    if _tf is None:
        try:
            import tensorflow as tf
            _tf = tf
        except Exception as e:
            raise RuntimeError("TensorFlow import failed: " + str(e))
    return _tf


class ModelWrapper:
    def __init__(self, model_path: Optional[str] = None,
                 token_to_idx: Optional[Dict[str, int]] = None,
                 idx_to_token: Optional[Dict[str, str]] = None,
                 max_len: Optional[int] = None):
        self.model_path = model_path
        self.model = None
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        self.max_len = max_len
        self.vocab_size = None

    def load_model(self, path: Optional[str] = None):
        path = path or self.model_path or DEFAULT_MODEL_PATH
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        tf = lazy_load_tf()
        try:
            self.model = tf.keras.models.load_model(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")
        # try infer max_len from model input shape
        if self.max_len is None:
            try:
                inp_shape = self.model.input_shape
                if isinstance(inp_shape, (list, tuple)):
                    inp_shape = inp_shape[0]
                if inp_shape and len(inp_shape) >= 2 and inp_shape[1] is not None:
                    self.max_len = int(inp_shape[1])
            except Exception:
                pass
        # infer vocab size from model output
        try:
            out_shape = self.model.output_shape
            if isinstance(out_shape, (list, tuple)):
                out_shape = out_shape[0]
            if out_shape and len(out_shape) >= 3:
                self.vocab_size = int(out_shape[2])
        except Exception:
            pass

    def load_mappings_from_files(self, token_to_idx_path: Optional[str], idx_to_token_path: Optional[str]):
        if token_to_idx_path and os.path.exists(token_to_idx_path):
            with open(token_to_idx_path, 'r', encoding='utf-8') as f:
                self.token_to_idx = json.load(f)
        if idx_to_token_path and os.path.exists(idx_to_token_path):
            with open(idx_to_token_path, 'r', encoding='utf-8') as f:
                self.idx_to_token = json.load(f)

    def try_auto_load_mappings_near_model(self):
        """Look for token_to_idx.json / idx_to_token.json next to the model file."""
        if not self.model_path:
            return
        base = os.path.dirname(self.model_path)
        cand1 = os.path.join(base, 'token_to_idx.json')
        cand2 = os.path.join(base, 'idx_to_token.json')
        if os.path.exists(cand1) and os.path.exists(cand2):
            self.load_mappings_from_files(cand1, cand2)

    def prepare_input(self, query: str):
        q = str(query)
        if not self.token_to_idx:
            # best-effort char mapping fallback (UNSAFE if model was trained differently)
            chars = sorted(list(set(q)))
            self.token_to_idx = {c: i + 1 for i, c in enumerate(chars)}
            self.idx_to_token = {str(i): c for c, i in self.token_to_idx.items()}
        indices = [self.token_to_idx.get(ch, self.token_to_idx.get('<UNK>', 0)) for ch in q]
        if self.max_len is None:
            self.max_len = max(1, len(indices))
        if len(indices) < self.max_len:
            indices = indices + [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return np.array([indices], dtype='int32')

    def decode_output_indices(self, indices) -> str:
        if self.idx_to_token is None:
            try:
                return ''.join([chr(i) for i in indices if i > 0])
            except Exception:
                return ''.join([str(i) for i in indices if i > 0])
        out = []
        for i in indices:
            key = str(int(i))
            token = None
            if isinstance(self.idx_to_token, dict):
                token = self.idx_to_token.get(key)
                if token is None:
                    # try int key
                    token = self.idx_to_token.get(int(i)) if int(i) in self.idx_to_token else None
            if token is None:
                token = ''
            out.append(token)
        return ''.join(out)

    def predict(self, query: str) -> Tuple[str, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        x = self.prepare_input(query)
        preds = self.model.predict(x)
        if isinstance(preds, list):
            preds = preds[0]
        preds = np.array(preds)
        if preds.ndim == 3:
            seq_indices = preds.argmax(axis=-1)[0]
            confidences = preds.max(axis=-1)[0]
            avg_conf = float(confidences.mean() * 100.0)
            decoded = self.decode_output_indices(seq_indices)
            return decoded, avg_conf
        elif preds.ndim == 2:
            seq_indices = preds[0].astype(int)
            decoded = self.decode_output_indices(seq_indices)
            return decoded, 100.0
        elif preds.ndim == 1:
            idx = int(preds.argmax())
            decoded = self.decode_output_indices([idx])
            return decoded, float(preds.max() * 100.0)
        else:
            return str(preds), 0.0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Medicine Name Reconstruction — LSTM Model")
        self.resize(900, 640)
        self.setMinimumSize(720, 520)

        # Theme
        self.palette_light = {
            "bg": "#F3F6FB", "card": "#FFFFFF", "fg": "#111827", "muted": "#6B7280",
            "primary": "#4F46E5", "success": "#10B981", "danger": "#EF4444",
            "border": "#E5E7EB", "accent": "#0B3B2E"
        }
        self.palette_dark = {
            "bg": "#0F172A", "card": "#111827", "fg": "#E5E7EB", "muted": "#9CA3AF",
            "primary": "#6366F1", "success": "#34D399", "danger": "#F87171",
            "border": "#1F2937", "accent": "#064E3B"
        }
        self.dark_mode = False

        # Model wrapper
        self.model_wrapper = ModelWrapper()
        # Try to pre-load default model (non-fatal)
        try:
            if os.path.exists(DEFAULT_MODEL_PATH):
                self.model_wrapper.model_path = DEFAULT_MODEL_PATH
                self.model_wrapper.load_model(DEFAULT_MODEL_PATH)
                self.model_wrapper.try_auto_load_mappings_near_model()
        except Exception as e:
            # ignore pre-load failures; user can load via menu
            print("Preload model failed:", e)

        # Menubar
        self._build_menubar()

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

        # Central UI
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)
        self.setCentralWidget(scroll)
        root = QVBoxLayout(container)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Header
        hdr = QVBoxLayout()
        self.lbl_title = QLabel("Medicine Name Reconstruction")
        self.lbl_title.setObjectName("Title")
        self.lbl_subtitle = QLabel("Model-only reconstruction UI — load your LSTM model and optional mappings.")
        self.lbl_subtitle.setWordWrap(True)
        self.lbl_subtitle.setObjectName("Subtitle")

        top_row = QHBoxLayout()
        top_row.addWidget(self.lbl_title, 1, Qt.AlignLeft)
        self.chk_dark = QCheckBox("Dark Mode")
        self.chk_dark.stateChanged.connect(self.toggle_dark_mode)
        top_row.addWidget(self.chk_dark, 0, Qt.AlignRight)

        hdr.addLayout(top_row)
        hdr.addWidget(self.lbl_subtitle)
        root.addLayout(hdr)

        # Thin top strip
        self.top_strip = QProgressBar()
        self.top_strip.setRange(0, 100)
        self.top_strip.setValue(100)
        self.top_strip.setTextVisible(False)
        self.top_strip.setObjectName("TopStrip")
        root.addWidget(self.top_strip)

        # Input card
        card_input = self._card()
        root.addWidget(card_input['frame'])
        in_layout = card_input['layout']

        in_layout.addWidget(self._section_label("Enter Medicine Name"))
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Enter medicine name (e.g., Paracetmol)")
        self.txt_input.returnPressed.connect(self.on_predict)
        in_layout.addWidget(self.txt_input)

        btn_row = QHBoxLayout()
        self.btn_predict = QPushButton("Run Model")
        self.btn_predict.clicked.connect(self.on_predict)
        self.btn_predict.setObjectName("PrimaryButton")
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_reset.setObjectName("SecondaryButton")
        btn_row.addWidget(self.btn_predict)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch(1)
        in_layout.addLayout(btn_row)

        # Model info row
        info_row = QHBoxLayout()
        self.lbl_model = QLabel("Model: (none)")
        self.lbl_model.setObjectName("Muted")
        info_row.addWidget(self.lbl_model)
        info_row.addStretch(1)
        self.btn_load_mappings = QPushButton("Load mappings...")
        self.btn_load_mappings.clicked.connect(self.load_mappings)
        info_row.addWidget(self.btn_load_mappings)
        in_layout.addLayout(info_row)

        # Output card
        card_out = self._card()
        root.addWidget(card_out['frame'])
        out_layout = card_out['layout']

        out_layout.addWidget(self._section_label("Predicted (Reconstructed) Name"))
        self.lbl_prediction = QLabel("—")
        self.lbl_prediction.setObjectName("Prediction")
        out_layout.addWidget(self.lbl_prediction)

        conf_row = QHBoxLayout()
        conf_row.addWidget(self._muted_label("Confidence"))
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setValue(0)
        self.meter.setTextVisible(False)
        self.meter.setObjectName("FlatBar")
        conf_row.addWidget(self.meter)
        self.lbl_conf = QLabel("")
        self.lbl_conf.setObjectName("Muted")
        conf_row.addWidget(self.lbl_conf)
        out_layout.addLayout(conf_row)

        # Details table
        self.details_frame = self._card()
        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Step", "Value"])
        self.tbl.horizontalHeader().setStretchLastSection(False)
        self.tbl.horizontalHeader().setDefaultSectionSize(320)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.setSelectionMode(QTableWidget.SingleSelection)
        self.details_frame['layout'].addWidget(self.tbl)
        self.details_frame['frame'].setVisible(True)
        root.addWidget(self.details_frame['frame'])

        # Apply theme
        self.apply_theme()

        # Update model label if preloaded
        if self.model_wrapper.model is not None:
            self.lbl_model.setText(f"Model: {os.path.basename(self.model_wrapper.model_path)}")
            self.status.showMessage("Model loaded (preload)", 3000)

    def _card(self):
        frame = QFrame()
        frame.setObjectName('Card')
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        return {'frame': frame, 'layout': layout}

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName('Section')
        return lbl

    def _muted_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName('Muted')
        return lbl

    def _build_menubar(self):
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        file_menu = menubar.addMenu('File')
        act_load_model = QAction('Load model...', self); act_load_model.triggered.connect(self._open_model)
        file_menu.addAction(act_load_model)
        act_open = QAction('Open dataset CSV...', self); act_open.triggered.connect(self._open_csv)
        file_menu.addAction(act_open)
        file_menu.addSeparator()
        act_exit = QAction('Exit', self); act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        help_menu = menubar.addMenu('Help')
        act_about = QAction('About', self); act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    def _about(self):
        QMessageBox.information(self, 'About',
                                'Medicine Name Reconstruction\nModel-only LSTM UI\nLoad your model and optional token mappings (JSON).')

    def _open_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Keras model', '', 'Keras model (*.keras);;All files (*)')
        if not path:
            return
        try:
            self.model_wrapper.model_path = path
            self.model_wrapper.load_model(path)
            self.model_wrapper.try_auto_load_mappings_near_model()
            self.lbl_model.setText(f"Model: {os.path.basename(path)}")
            self.status.showMessage(f"Loaded model: {os.path.basename(path)}", 3000)
        except Exception as e:
            QMessageBox.critical(self, 'Model load error', f"Failed to load model:\n{e}")

    def load_mappings(self):
        tpath, _ = QFileDialog.getOpenFileName(self, 'Load token_to_idx.json', '', 'JSON files (*.json);;All files (*)')
        if not tpath:
            return
        ipath, _ = QFileDialog.getOpenFileName(self, 'Load idx_to_token.json', '', 'JSON files (*.json);;All files (*)')
        if not ipath:
            return
        try:
            self.model_wrapper.load_mappings_from_files(tpath, ipath)
            self.status.showMessage('Mappings loaded', 3000)
        except Exception as e:
            QMessageBox.critical(self, 'Mappings error', f"Failed to load mappings:\n{e}")

    def _open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open dataset CSV', '', 'CSV Files (*.csv);;All files (*)')
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            if not {'Input', 'Output'}.issubset(df.columns):
                raise ValueError('CSV must contain columns named exactly: Input, Output')
            self.status.showMessage(f'Loaded dataset: {os.path.basename(path)}', 3000)
        except Exception as e:
            QMessageBox.critical(self, 'Read error', f'Failed to read CSV:\n{e}')

    def toggle_dark_mode(self, _state):
        self.dark_mode = self.chk_dark.isChecked()
        self.apply_theme()

    def apply_theme(self):
        pal = self.palette_dark if self.dark_mode else self.palette_light
        ss = f"""
            QWidget {{ background: {pal['bg']}; color: {pal['fg']}; }}
            QMenuBar {{ background: {pal['card']}; }}
            QStatusBar {{ background: {pal['bg']}; color: {pal['muted']}; }}

            QFrame#Card {{
                background: {pal['card']};
                border: 1px solid {pal['border']};
                border-radius: 8px;
            }}

            QLabel#Title {{ font-size: 22px; font-weight: 700; color: {pal['fg']}; }}
            QLabel#Subtitle {{ font-size: 12px; color: {pal['muted']}; }}
            QLabel#Section {{ font-size: 13px; font-weight: 700; color: {pal['fg']}; }}
            QLabel#Prediction {{ font-size: 14px; font-weight: 700; color: {pal['fg']}; }}
            QLabel#Muted {{ color: {pal['muted']}; }}

            QPushButton#PrimaryButton {{
                background: {pal['primary']};
                color: white; border: none; border-radius: 6px; padding: 8px 14px;
                font-weight: 600;
            }}
            QPushButton#SecondaryButton {{
                background: {pal['border']};
                color: {pal['fg']}; border: none; border-radius: 6px; padding: 8px 14px;
            }}
            QPushButton#GhostButton {{
                background: transparent; color: {pal['muted']};
                border: none; padding: 6px 8px;
            }}

            QLineEdit {{
                background: {pal['card']}; color: {pal['fg']};
                border: 1px solid {pal['border']}; border-radius: 6px; padding: 8px;
            }}
            QLineEdit:focus {{ border: 1px solid {pal['primary']}; }}

            QHeaderView::section {{
                background: {pal['card']}; color: {pal['muted']};
                border: 1px solid {pal['border']}; padding: 6px;
                font-weight: 600;
            }}
            QTableWidget {{
                gridline-color: {pal['border']};
                background: {pal['card']}; color: {pal['fg']};
                selection-background-color: {pal['primary']};
                selection-color: white;
            }}

            QProgressBar#FlatBar {{
                border: 1px solid {pal['border']}; border-radius: 6px; background: {pal['border']};
                height: 16px;
            }}
            QProgressBar#FlatBar::chunk {{ background: {pal['primary']}; border-radius: 6px; }}
            QProgressBar#TopStrip {{
                background: {pal['border']}; height: 8px; border: 1px solid {pal['border']};
            }}
            QProgressBar#TopStrip::chunk {{ background: {pal['primary']}; }}
        """
        self.setStyleSheet(ss)
        self.status.showMessage("Dark Mode On" if self.dark_mode else "Ready", 2000)

    def on_predict(self):
        query = self.txt_input.text().strip()
        if not query:
            QMessageBox.warning(self, 'Input needed', 'Please enter a medicine name.')
            return
        if self.model_wrapper.model is None:
            QMessageBox.warning(self, 'No model', 'No model loaded. Use File -> Load model...')
            return
        try:
            pred, score = self.model_wrapper.predict(query)
        except Exception as e:
            QMessageBox.critical(self, 'Prediction error', f'Could not predict:\n{e}')
            return
        self.meter.setValue(int(max(0.0, min(100.0, score))))
        self.lbl_conf.setText(f"{score:.1f}")

        # Fill details table with some helpful debug rows
        self.tbl.setRowCount(0)
        rows = [
            ("Input", query),
            ("Decoded prediction", pred),
            ("Confidence", f"{score:.1f}%"),
            ("Model path", str(self.model_wrapper.model_path or '')),
            ("Max len", str(self.model_wrapper.max_len or 'unknown')),
            ("Vocab size", str(self.model_wrapper.vocab_size or 'unknown'))
        ]
        for k, v in rows:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QTableWidgetItem(str(k)))
            self.tbl.setItem(r, 1, QTableWidgetItem(str(v)))

        # Predicted medicine shown with underline & danger color
        red_color = self.palette_dark['danger'] if self.dark_mode else self.palette_light['danger']
        safe_pred = (
            str(pred)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if score < MIN_CONFIDENCE:
            html = (
                "Closest: "
                f"<span style='color:{red_color}; text-decoration: underline;'>"
                f"{safe_pred}</span>"
                f"  (low confidence {score:.1f})"
            )
            self.lbl_prediction.setText(html)
            self.status.showMessage('Closest match (low confidence)', 3000)
        else:
            html = (
                "Prediction: "
                f"<span style='color:{red_color}; text-decoration: underline;'>"
                f"{safe_pred}</span>"
            )
            self.lbl_prediction.setText(html)
            self.status.showMessage('Prediction ready', 2000)

    def on_reset(self):
        self.txt_input.clear()
        self.lbl_prediction.setText('—')
        self.meter.setValue(0)
        self.lbl_conf.setText('')
        self.tbl.setRowCount(0)
        self.status.showMessage('Cleared', 1500)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
