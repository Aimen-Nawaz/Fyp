"""
Medicine name reconstruction prediction logic
"""
import re
import numpy as np
import Levenshtein
from typing import Dict, List, Tuple


def clean_raw(raw: str) -> str:
    """Remove trailing repeated characters"""
    if not raw:
        return raw
    cleaned = re.sub(r'(.)\1{2,}$', r'\1', raw)
    while len(cleaned) >= 2 and cleaned[-1] == cleaned[-2]:
        cleaned = cleaned[:-1]
    return cleaned.strip()


def neural_predict(text: str, model, token_to_idx: Dict, idx_to_token: Dict, config: Dict) -> Tuple[str, float]:
    """
    Predict medicine name using neural network
    Returns: (predicted_text, confidence)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    MAX_LEN = config['max_len']
    PAD_IDX = config['pad_idx']
    
    # Encode text
    enc = [token_to_idx.get(c, PAD_IDX) for c in list(text.lower().strip())]
    padded = pad_sequences([enc], maxlen=MAX_LEN, padding='post', value=PAD_IDX, truncating='post')
    
    # Get predictions
    probs = model.predict(padded, verbose=0)[0]
    idx = np.argmax(probs, axis=-1)
    
    # Decode output
    input_len = len(text.strip())
    max_out_len = min(input_len + 3, MAX_LEN)
    chars, confs = [], []
    
    for t in range(max_out_len):
        tok = idx_to_token.get(int(idx[t]), '')
        p = float(probs[t, idx[t]])
        if tok in ('<EOS>', '<PAD>', ''):
            break
        if p < 0.30 and t >= input_len:
            break
        chars.append(tok)
        confs.append(p)
    
    return ''.join(chars), (float(np.mean(confs)) if confs else 0.0)


def snap_to_real_top_k(query: str, raw_pred: str, known_names: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """
    Find closest real medicine names using Levenshtein distance
    Returns: list of (name, snap_score) tuples sorted by score
    """
    candidates = []
    for candidate in known_names:
        if abs(len(candidate) - len(query)) > 4:
            continue
        d_input = Levenshtein.distance(query, candidate)
        d_raw = Levenshtein.distance(raw_pred, candidate) if raw_pred else 99
        d_min = min(d_input, d_raw)
        snap_score = 1.0 / (1 + d_min)
        candidates.append((candidate, snap_score))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:k]


def predict_one(text: str, model_artifacts: Dict) -> Dict:
    """
    Single prediction with alternatives
    Returns: {prediction, confidence, alternatives}
    """
    text = text.lower().strip()
    if not text:
        return {'prediction': '', 'confidence': 0.0, 'alternatives': []}
    
    model = model_artifacts['model']
    token_to_idx = model_artifacts['token_to_idx']
    idx_to_token = model_artifacts['idx_to_token']
    config = model_artifacts['config']
    known_names = model_artifacts['known_names']
    
    # Get neural prediction
    raw, neural_conf = neural_predict(text, model, token_to_idx, idx_to_token, config)
    cleaned = clean_raw(raw)
    
    # Find closest real medicine names
    top_matches = snap_to_real_top_k(text, cleaned, known_names, k=10)
    
    if not top_matches:
        return {'prediction': text, 'confidence': 0.0, 'alternatives': []}
    
    # Best match
    best_name, best_snap = top_matches[0]
    best_conf = round(0.7 * best_snap + 0.3 * neural_conf, 4)
    
    # Alternatives
    alternatives = []
    for match_name, snap_score in top_matches[1:]:
        conf = round(0.7 * snap_score + 0.3 * neural_conf, 4)
        alternatives.append({'prediction': match_name, 'confidence': conf})
    
    return {
        'prediction': best_name,
        'confidence': best_conf,
        'alternatives': alternatives
    }


def predict_batch(texts: List[str], model_artifacts: Dict) -> List[Dict]:
    """
    Batch prediction
    Returns: list of prediction results
    """
    results = []
    for text in texts:
        text = str(text).strip()
        if text:
            result = predict_one(text, model_artifacts)
            result['input'] = text
            results.append(result)
    return results
