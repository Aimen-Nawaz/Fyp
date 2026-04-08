 #Features:
# - Loads a Keras (.keras / SavedModel) seq2seq or classification model
# - Tries to auto-detect token mappings (token_to_idx / idx_to_token) from JSON files or user selection
# 
# Changes:
#   • Removed "Not found" error popup
#   • Always show top matches in details
#   • Details panel opened by default; label shows Closest for low confidence
import os
import sys
from typing import Tuple

import sys
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import jellyfish
from unidecode import unidecode

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QProgressBar, QTableWidget, QTableWidgetItem,
    QMessageBox, QScrollArea, QCheckBox, QStatusBar, QMenuBar, QFileDialog, QFrame
)

DATA_PATH = "medicine_names_dataset.csv"   # expects columns: Input, Output
MIN_CONFIDENCE = 65.0                      # still used for accuracy display only

# ------------------------ Core helpers (no TF) ------------------------
def normalize_text(s: str) -> str:
    s = str(s)
    s = unidecode(s)
    s = s.lower().strip()
    s = " ".join(s.split())
    return s

def safe_metaphone(x: str) -> str:
    try:
        return jellyfish.metaphone(x or "")
    except Exception:
        return ""

def safe_double_metaphone_primary(x: str) -> str:
    try:
        dm = getattr(jellyfish, "double_metaphone", None)
        if callable(dm):
            res = dm(x or "")
            return (res[0] or "") if isinstance(res, (tuple, list)) else str(res or "")
        else:
            return safe_metaphone(x)
    except Exception:
        return safe_metaphone(x)

def build_candidate_table(df: pd.DataFrame) -> pd.DataFrame:
    outs = pd.Series(df["Output"].astype(str).unique(), name="candidate")
    norm = outs.apply(normalize_text)
    meta1 = norm.apply(safe_metaphone)
    meta2 = norm.apply(safe_double_metaphone_primary)
    return pd.DataFrame(
        {"candidate": outs.values, "norm": norm.values,
         "metaphone": meta1.values, "dmetaphone": meta2.values}
    )

def jw_0_100(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        return 100.0 * jellyfish.jaro_winkler_similarity(a, b)
    except Exception:
        return 0.0

def similarity_score(query: str, row) -> float:
    qn = normalize_text(query)
    cn = row["norm"]
    r_base = fuzz.WRatio(qn, cn)
    r_part = fuzz.partial_ratio(qn, cn)
    r_token = fuzz.token_sort_ratio(qn, cn)
    q_meta1 = safe_metaphone(qn)
    q_meta2 = safe_double_metaphone_primary(qn)
    p1 = jw_0_100(q_meta1, row["metaphone"])
    p2 = jw_0_100(q_meta2, row["dmetaphone"])
    len_q = max(1, len(qn)); len_c = max(1, len(cn))
    length_ratio = min(len_q, len_c) / max(len_q, len_c)
    length_boost = 10 * (length_ratio - 0.5)  # ~ -5..+5
    score = 0.40*r_base + 0.20*r_part + 0.20*r_token + 0.15*max(p1, p2) + length_boost
    return float(score)

def predict_best(query: str, candidates_df: pd.DataFrame, top_k: int = 10) -> Tuple[str, float, pd.DataFrame]:
    """Return (best_candidate, best_score, topk_df[candidate, score])"""
    if not query.strip():
        return "", 0.0, pd.DataFrame(columns=["candidate", "score"])
    scores = candidates_df.apply(lambda r: similarity_score(query, r), axis=1)
    order = np.argsort(-scores.values)
    k = min(top_k, len(order))
    top_idx = order[:k]
    top_df = candidates_df.iloc[top_idx][["candidate"]].copy()
    top_df["score"] = scores.iloc[top_idx].values
    best_row = top_df.iloc[0]
    return str(best_row["candidate"]), float(best_row["score"]), top_df.reset_index(drop=True)

def compute_accuracy(df: pd.DataFrame, candidates_df: pd.DataFrame) -> Tuple[float, int]:
    """Accuracy only counts matches with score >= MIN_CONFIDENCE."""
    test_df = df[["Input", "Output"]].dropna().astype(str)
    total = len(test_df)
    if total == 0:
        return 0.0, 0
    correct = 0
    for _, row in test_df.iterrows():
        pred, score, _ = predict_best(row["Input"], candidates_df, top_k=1)
        if score >= MIN_CONFIDENCE and pred == row["Output"]:
            correct += 1
    return correct/total, total

# ------------------------ Main Window ------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Medicine Name Reconstruction")
        self.resize(900, 640)
        self.setMinimumSize(720, 520)

        # Theme palettes (OCR-style)
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

        # Data
        df = self._load_dataset_or_die()
        if df is None:
            return
        self.df = df.astype({"Input": str, "Output": str})
        try:
            self.candidates = build_candidate_table(self.df)
        except Exception as e:
            QMessageBox.critical(self, "Init error", f"Failed to build candidates:\n{e}")
            QTimer.singleShot(0, self.close)
            return

        # Menubar
        self._build_menubar()

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

        # Scrollable central widget
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
        self.lbl_subtitle = QLabel("Fix misspelled or partial medicine names .")
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

        # Thin top progress strip (visual)
        self.top_strip = QProgressBar()
        self.top_strip.setRange(0, 100)
        self.top_strip.setValue(100)
        self.top_strip.setTextVisible(False)
        self.top_strip.setObjectName("TopStrip")
        root.addWidget(self.top_strip)

        # Card: Input
        card_input = self._card()
        root.addWidget(card_input["frame"])
        in_layout = card_input["layout"]

        in_layout.addWidget(self._section_label("Enter Medicine Name"))
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Enter medicine name (e.g., Paracetmol)")
        self.txt_input.returnPressed.connect(self.on_predict)
        in_layout.addWidget(self.txt_input)

        btn_row = QHBoxLayout()
        self.btn_predict = QPushButton("Start AI Analysis")
        self.btn_predict.clicked.connect(self.on_predict)
        self.btn_predict.setObjectName("PrimaryButton")
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_reset.setObjectName("SecondaryButton")
        btn_row.addWidget(self.btn_predict)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch(1)
        in_layout.addLayout(btn_row)

        # Card: Output
        card_out = self._card()
        root.addWidget(card_out["frame"])
        out_layout = card_out["layout"]

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

        # Details (top matches table) — open by default
        self.btn_toggle = QPushButton("Hide details ▾")
        self.btn_toggle.setObjectName("GhostButton")
        self.btn_toggle.clicked.connect(self._toggle_details)
        root.addWidget(self.btn_toggle, alignment=Qt.AlignLeft)

        self.details_frame = self._card()
        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Candidate", "Score"])
        self.tbl.horizontalHeader().setStretchLastSection(False)
        self.tbl.horizontalHeader().setDefaultSectionSize(320)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.setSelectionMode(QTableWidget.SingleSelection)
        self.details_frame["layout"].addWidget(self.tbl)
        self.details_frame["frame"].setVisible(True)  # show by default
        root.addWidget(self.details_frame["frame"])

        # Card: Accuracy
        card_acc = self._card()
        root.addWidget(card_acc["frame"])
        acc_layout = card_acc["layout"]
        acc_layout.addWidget(self._section_label("Model Accuracy"))
        self.lbl_acc = QLabel("Evaluating…")
        self.lbl_acc.setObjectName("Muted")
        acc_layout.addWidget(self.lbl_acc)
        self.acc_bar = QProgressBar()
        self.acc_bar.setRange(0, 0)  # indeterminate
        self.acc_bar.setTextVisible(False)
        self.acc_bar.setObjectName("FlatBar")
        acc_layout.addWidget(self.acc_bar)

        # Apply theme
        self.apply_theme()

        # Compute accuracy shortly
        QTimer.singleShot(120, self.compute_accuracy_async)

    # ---- helpers ----
    def _card(self):
        frame = QFrame()
        frame.setObjectName("Card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        return {"frame": frame, "layout": layout}

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("Section")
        return lbl

    def _muted_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("Muted")
        return lbl

    # ---- Menu ----
    def _build_menubar(self):
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        file_menu = menubar.addMenu("File")
        act_reset = QAction("Reset", self); act_reset.triggered.connect(self.on_reset)
        file_menu.addAction(act_reset)
        file_menu.addSeparator()
        act_open = QAction("Open CSV...", self); act_open.triggered.connect(self._open_csv)
        file_menu.addAction(act_open)
        file_menu.addSeparator()
        act_exit = QAction("Exit", self); act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        help_menu = menubar.addMenu("Help")
        act_about = QAction("About", self); act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    def _about(self):
        QMessageBox.information(
            self, "About",
            "Medicine Name Reconstruction\n"
            "Tensorflow"
            "PySide6 UI — OCR-style theme"
        )

    def _open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path).astype({"Input": str, "Output": str})
            if not {"Input", "Output"}.issubset(df.columns):
                raise ValueError("CSV must contain columns named exactly: Input, Output")
            self.df = df
            self.candidates = build_candidate_table(self.df)
            self.status.showMessage(f"Loaded dataset: {os.path.basename(path)}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Read error", f"Failed to read CSV:\n{e}")

    # ---- Theme ----
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

    # ---- Dataset loader ----
    def _load_dataset_or_die(self) -> pd.DataFrame | None:
        if not os.path.exists(DATA_PATH):
            QMessageBox.critical(
                self, "Missing dataset",
                f"Cannot find `{DATA_PATH}`.\n\nMake sure the CSV exists and has columns: Input, Output."
            )
            QTimer.singleShot(0, self.close)
            return None
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception as e:
            QMessageBox.critical(self, "Read error", f"Failed to read CSV:\n{e}")
            QTimer.singleShot(0, self.close)
            return None
        if not {"Input", "Output"}.issubset(df.columns):
            QMessageBox.critical(self, "Bad dataset", "CSV must contain columns named exactly: Input, Output")
            QTimer.singleShot(0, self.close)
            return None
        return df

    # ---- Interactions ----
    def _toggle_details(self):
        vis = self.details_frame["frame"].isVisible()
        self.details_frame["frame"].setVisible(not vis)
        self.btn_toggle.setText("Hide details ▾" if not vis else "Show details ▸")

    def on_predict(self):
        query = self.txt_input.text().strip()
        if not query:
            # keep a light warning so users know why nothing happened
            QMessageBox.warning(self, "Input needed", "Please enter a medicine name.")
            return
        try:
            pred, score, topk = predict_best(query, self.candidates, top_k=10)
        except Exception as e:
            QMessageBox.critical(self, "Prediction error", f"Could not predict:\n{e}")
            return

        # Confidence bar + number
        self.meter.setValue(int(max(0.0, min(100.0, score))))
        self.lbl_conf.setText(f"{score:.1f}")

        # Always fill details with top matches
        self.tbl.setRowCount(0)
        for _, row in topk.iterrows():
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QTableWidgetItem(str(row["candidate"])))
            self.tbl.setItem(r, 1, QTableWidgetItem(f"{row['score']:.1f}"))

        # Keep details visible
        if not self.details_frame["frame"].isVisible():
            self._toggle_details()

        # Label text depending on confidence (NO popup)
        
        safe_pred = (
            str(pred)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        red_color = (
            self.palette_dark["danger"]
            if self.dark_mode
            else self.palette_light["danger"]
        )

        if score < MIN_CONFIDENCE:
            html = (
                "Closest: "
                f"<span style='color:{red_color}; text-decoration: underline;'>"
                f"{safe_pred}</span>"
                f"  (low confidence {score:.1f})"
            )
            self.lbl_prediction.setText(html)
            self.status.showMessage("Closest match (low confidence)", 3000)
        else:
            html = (
                "Prediction: "
                f"<span style='color:{red_color}; text-decoration: underline;'>"
                f"{safe_pred}</span>"
            )
            self.lbl_prediction.setText(html)
            self.status.showMessage("Prediction ready", 2000)

    def on_reset(self):
        self.txt_input.clear()
        self.lbl_prediction.setText("—")
        self.meter.setValue(0)
        self.lbl_conf.setText("")
        self.tbl.setRowCount(0)
        self.status.showMessage("Cleared", 1500)

    def compute_accuracy_async(self):
        try:
            acc, n = compute_accuracy(self.df, self.candidates)
            self.lbl_acc.setText(
                f"Model Accuracy: {acc*100:.2f}% (Evaluated on {n} samples, threshold={MIN_CONFIDENCE:.0f})"
            )
        except Exception:
            self.lbl_acc.setText("Model Accuracy: could not compute")
        finally:
            self.acc_bar.setRange(0, 100)
            self.acc_bar.setValue(100)
            self.status.showMessage("Ready", 2000)

# ------------------------ Entry ------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
