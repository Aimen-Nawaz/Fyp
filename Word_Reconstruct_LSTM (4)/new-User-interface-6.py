import sys
import os
import json
import numpy as np
#import tensorflow as tf

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox,
    QProgressBar, QHBoxLayout
)
from PySide6.QtCore import Qt


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "Word_Reconstruct_LSTM.keras")
DEFAULT_TOKEN_TO_IDX = os.path.join(BASE_DIR, "token_to_idx.json")
DEFAULT_IDX_TO_TOKEN = os.path.join(BASE_DIR, "idx_to_token.json")


# =========================
# MODEL WRAPPER
# =========================
class ModelWrapper:
    def __init__(self):
        self.model = None
        self.token_to_idx = None
        self.idx_to_token = None
        self.max_len = 30

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def load_mappings(self, token_path, idx_path):
        with open(token_path, "r") as f:
            self.token_to_idx = json.load(f)
        with open(idx_path, "r") as f:
            self.idx_to_token = json.load(f)

    def _fallback_mapping(self, text):
        chars = sorted(set(text))
        self.token_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx_to_token = {i + 1: c for i, c in enumerate(chars)}

    def encode(self, text):
        if not self.token_to_idx:
            self._fallback_mapping(text)

        seq = [self.token_to_idx.get(c, 0) for c in text.lower()]
        seq = seq[:self.max_len]
        seq += [0] * (self.max_len - len(seq))
        return np.array(seq).reshape(1, -1)

    def decode(self, seq):
        return "".join(self.idx_to_token.get(int(i), "") for i in seq)

    def predict(self, text):
        if self.model is None:
            raise RuntimeError("Model not loaded")

        encoded = self.encode(text)
        preds = self.model.predict(encoded, verbose=0)
        pred_seq = np.argmax(preds, axis=-1)[0]

        result = self.decode(pred_seq)
        confidence = float(np.max(preds))

        return result.strip(), confidence


# =========================
# MAIN WINDOW
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medicine Name Reconstruction — LSTM")
        self.resize(900, 500)

        self.model_wrapper = ModelWrapper()
        self._init_ui()
        self._auto_load()

    def _init_ui(self):
        central = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Medicine Name Reconstruction")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Enter incorrect medicine name (e.g. pandl)")
        layout.addWidget(self.input_box)

        btn_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Model")
        self.run_btn.clicked.connect(self.run_model)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)

        self.load_map_btn = QPushButton("Load Mappings")
        self.load_map_btn.clicked.connect(self.load_mappings)

        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.load_model_btn)
        btn_layout.addWidget(self.load_map_btn)

        layout.addLayout(btn_layout)

        self.model_label = QLabel("Model: (none)")
        layout.addWidget(self.model_label)

        self.output_label = QLabel("Predicted Name: —")
        self.output_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.output_label)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        layout.addWidget(self.conf_bar)

        central.setLayout(layout)
        self.setCentralWidget(central)

    # =========================
    # AUTO LOAD
    # =========================
    def _auto_load(self):
        if os.path.exists(DEFAULT_MODEL_PATH):
            try:
                self.model_wrapper.load_model(DEFAULT_MODEL_PATH)
                self.model_label.setText(f"Model: {os.path.basename(DEFAULT_MODEL_PATH)}")
            except Exception as e:
                QMessageBox.warning(self, "Model Error", str(e))

        if os.path.exists(DEFAULT_TOKEN_TO_IDX) and os.path.exists(DEFAULT_IDX_TO_TOKEN):
            try:
                self.model_wrapper.load_mappings(DEFAULT_TOKEN_TO_IDX, DEFAULT_IDX_TO_TOKEN)
            except:
                pass

    # =========================
    # BUTTON ACTIONS
    # =========================
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Keras Model", "", "Keras Model (*.keras)"
        )
        if path:
            try:
                self.model_wrapper.load_model(path)
                self.model_label.setText(f"Model: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def load_mappings(self):
        token_path, _ = QFileDialog.getOpenFileName(self, "Load token_to_idx.json")
        idx_path, _ = QFileDialog.getOpenFileName(self, "Load idx_to_token.json")

        if token_path and idx_path:
            try:
                self.model_wrapper.load_mappings(token_path, idx_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def run_model(self):
        text = self.input_box.text().strip()
        if not text:
            return

        try:
            result, conf = self.model_wrapper.predict(text)
            self.output_label.setText(f"Predicted Name: {result}")
            self.conf_bar.setValue(int(conf * 100))
        except Exception as e:
            QMessageBox.warning(self, "No model", str(e))


# =========================
# APP START
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
