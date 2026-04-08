# Changes:
#   • Added model integration display (UI only, no logic change)

import os
import sys
from typing import Tuple

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

DATA_PATH = "medicine_names_dataset.csv"
MIN_CONFIDENCE = 65.0

# ------------------------ Core helpers ------------------------
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
    length_boost = 10 * (length_ratio - 0.5)
    score = 0.40*r_base + 0.20*r_part + 0.20*r_token + 0.15*max(p1, p2) + length_boost
    return float(score)

def predict_best(query: str, candidates_df: pd.DataFrame, top_k: int = 10) -> Tuple[str, float, pd.DataFrame]:
    if not query.strip():
        return "", 0.0, pd.DataFrame(columns=["candidate", "score"])
    scores = candidates_df.apply(lambda r: similarity_score(query, r), axis=1)
    order = np.argsort(-scores.values)
    top_idx = order[:top_k]
    top_df = candidates_df.iloc[top_idx][["candidate"]].copy()
    top_df["score"] = scores.iloc[top_idx].values
    best_row = top_df.iloc[0]
    return str(best_row["candidate"]), float(best_row["score"]), top_df.reset_index(drop=True)

# ------------------------ Main Window ------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Medicine Name Reconstruction")

        df = pd.read_csv(DATA_PATH).astype({"Input": str, "Output": str})
        self.df = df
        self.candidates = build_candidate_table(self.df)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")
        self.status.showMessage("Model integrated successfully (Fuzzy Engine)", 3000)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)
        self.setCentralWidget(scroll)
        root = QVBoxLayout(container)

        # Header
        self.lbl_title = QLabel("Medicine Name Reconstruction")
        root.addWidget(self.lbl_title)

        # ✅ MODEL LABEL ADDED HERE
        self.lbl_model = QLabel("Model: Integrated (Fuzzy + Phonetic Engine)")
        root.addWidget(self.lbl_model)

        # Input
        self.txt_input = QLineEdit()
        root.addWidget(self.txt_input)

        self.btn_predict = QPushButton("Start AI Analysis")
        self.btn_predict.clicked.connect(self.on_predict)
        root.addWidget(self.btn_predict)

        # Output
        self.lbl_prediction = QLabel("—")
        root.addWidget(self.lbl_prediction)

        # ✅ MODEL TYPE BELOW PREDICTION
        self.lbl_model_type = QLabel("Engine: Fuzzy + Phonetic Matcher")
        root.addWidget(self.lbl_model_type)

        self.meter = QProgressBar()
        root.addWidget(self.meter)

        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Candidate", "Score"])
        root.addWidget(self.tbl)

    def on_predict(self):
        query = self.txt_input.text().strip()

        # ✅ MODEL RUNNING STATUS
        self.status.showMessage("Running model inference...", 1000)

        pred, score, topk = predict_best(query, self.candidates, top_k=10)

        self.lbl_prediction.setText(f"Prediction: {pred}")
        self.meter.setValue(int(score))

        self.tbl.setRowCount(0)
        for _, row in topk.iterrows():
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QTableWidgetItem(str(row["candidate"])))
            self.tbl.setItem(r, 1, QTableWidgetItem(f"{row['score']:.1f}"))

    def _about(self):
        QMessageBox.information(
            self, "About",
            "Medicine Name Reconstruction\n"
            "Model: Fuzzy Matching + Phonetic Encoding\n"
            "No Deep Learning (Lightweight AI)\n"
            "PySide6 UI"
        )

# ------------------------ Entry ------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()