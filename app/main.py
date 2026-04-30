"""
Medicine Name Reconstruction — Local Web App
Run:  python app.py  →  http://localhost:5000
"""
import os, sys, json, time, logging, re
import numpy as np
import Levenshtein
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, '..', 'model_artifacts')

app = Flask(__name__)
CORS(app)

model = token_to_idx = idx_to_token = config = known_names = None


def load_artifacts():
    global model, token_to_idx, idx_to_token, config, known_names

    paths = {
        'config': os.path.join(ARTIFACTS_DIR, 'model_config.json'),
        't2i'   : os.path.join(ARTIFACTS_DIR, 'token_to_idx.json'),
        'i2t'   : os.path.join(ARTIFACTS_DIR, 'idx_to_token.json'),
        'model' : os.path.join(ARTIFACTS_DIR, 'medicine_lstm.keras'),
        'names' : os.path.join(ARTIFACTS_DIR, 'known_names.txt'),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        log.error("Missing files: %s", missing)
        log.error("Download from Kaggle and place in: %s", os.path.abspath(ARTIFACTS_DIR))
        sys.exit(1)

    with open(paths['config']) as f: config = json.load(f)
    with open(paths['t2i'])    as f: token_to_idx = json.load(f)
    with open(paths['i2t'])    as f:
        idx_to_token = {int(k): v for k, v in json.load(f).items()}
    with open(paths['names'], encoding='utf-8') as f:
        known_names = sorted(set(line.strip().lower() for line in f if line.strip()))
    log.info("Loaded %d known names", len(known_names))

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.optimizers import Adam
    PAD_IDX = config['pad_idx']

    @tf.function
    def masked_sparse_ce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        loss   = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        mask   = tf.cast(tf.not_equal(y_true, PAD_IDX), tf.float32)
        return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)

    class MaskedAccuracy(tf.keras.metrics.Metric):
        def __init__(self, pad_idx=PAD_IDX, name='masked_accuracy', **kw):
            super().__init__(name=name, **kw)
            self.pad_idx = pad_idx
            self.correct = self.add_weight(name='correct', initializer='zeros')
            self.total   = self.add_weight(name='total',   initializer='zeros')
        def update_state(self, yt, yp, sw=None):
            yt = tf.cast(tf.reshape(yt, [-1]), tf.int32)
            yp = tf.cast(tf.argmax(yp, axis=-1), tf.int32); yp = tf.reshape(yp, [-1])
            m = tf.not_equal(yt, self.pad_idx)
            self.correct.assign_add(tf.reduce_sum(tf.cast(
                tf.equal(tf.boolean_mask(yt, m), tf.boolean_mask(yp, m)), tf.float32)))
            self.total.assign_add(tf.cast(tf.reduce_sum(tf.cast(m, tf.int32)), tf.float32))
        def result(self):      return self.correct / (self.total + 1e-8)
        def reset_state(self): self.correct.assign(0.); self.total.assign(0.)

    model = tf.keras.models.load_model(
        paths['model'],
        custom_objects={'masked_sparse_ce': masked_sparse_ce, 'MaskedAccuracy': MaskedAccuracy},
        compile=False,
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss=masked_sparse_ce,
                  metrics=[MaskedAccuracy(PAD_IDX)])
    log.info("Model loaded")


def clean_raw(raw):
    if not raw: return raw
    cleaned = re.sub(r'(.)\1{2,}$', r'\1', raw)
    while len(cleaned) >= 2 and cleaned[-1] == cleaned[-2]:
        cleaned = cleaned[:-1]
    return cleaned.strip()


def neural_predict(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    MAX_LEN = config['max_len']; PAD_IDX = config['pad_idx']
    enc    = [token_to_idx.get(c, PAD_IDX) for c in list(text.lower().strip())]
    padded = pad_sequences([enc], maxlen=MAX_LEN, padding='post', value=PAD_IDX, truncating='post')
    probs  = model.predict(padded, verbose=0)[0]
    idx    = np.argmax(probs, axis=-1)
    input_len   = len(text.strip())
    max_out_len = min(input_len + 3, MAX_LEN)
    chars, confs = [], []
    for t in range(max_out_len):
        tok = idx_to_token.get(int(idx[t]), '')
        p   = float(probs[t, idx[t]])
        if tok in ('<EOS>','<PAD>',''): break
        if p < 0.30 and t >= input_len: break
        chars.append(tok); confs.append(p)
    return ''.join(chars), (float(np.mean(confs)) if confs else 0.0)


def snap_to_real_top_k(query, raw_pred, k=10):
    candidates = []
    for candidate in known_names:
        if abs(len(candidate) - len(query)) > 4: continue
        d_input = Levenshtein.distance(query, candidate)
        d_raw   = Levenshtein.distance(raw_pred, candidate) if raw_pred else 99
        d_min   = min(d_input, d_raw)
        snap_score = 1.0 / (1 + d_min)
        candidates.append((candidate, snap_score))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:k]


def predict_one(text):
    text = text.lower().strip()
    if not text:
        return {'prediction': '', 'confidence': 0.0, 'alternatives': []}
    raw, neural_conf = neural_predict(text)
    cleaned = clean_raw(raw)
    top_matches = snap_to_real_top_k(text, cleaned, k=10)
    
    if not top_matches:
        return {'prediction': text, 'confidence': 0.0, 'alternatives': []}
        
    best_name, best_snap = top_matches[0]
    best_conf = round(0.7 * best_snap + 0.3 * neural_conf, 4)
    
    alternatives = []
    for match_name, snap_score in top_matches[1:]:
        conf = round(0.7 * snap_score + 0.3 * neural_conf, 4)
        alternatives.append({'prediction': match_name, 'confidence': conf})
        
    return {'prediction': best_name, 'confidence': best_conf, 'alternatives': alternatives}


HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>💊 Medicine Name Reconstruction</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:linear-gradient(135deg,#EEF2FF,#F0FDFA);min-height:100vh;
       display:flex;flex-direction:column;align-items:center;padding:32px 16px;color:#1E293B}
  .hero{background:linear-gradient(135deg,#4F46E5,#10B981);border-radius:20px;
        padding:28px 36px;max-width:680px;width:100%;color:white;margin-bottom:22px;
        box-shadow:0 10px 40px rgba(79,70,229,.3)}
  .hero h1{font-size:26px;font-weight:800;margin-bottom:4px}
  .hero p{opacity:.9;font-size:14px}
  .card{background:white;border-radius:16px;padding:26px 30px;max-width:680px;width:100%;
        box-shadow:0 4px 20px rgba(0,0,0,.08);margin-bottom:18px}
  .card h2{font-size:11px;font-weight:700;color:#4F46E5;text-transform:uppercase;
           letter-spacing:.12em;margin-bottom:16px}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  input[type=text]{flex:1;min-width:220px;padding:13px 16px;border:2px solid #E2E8F0;
                   border-radius:12px;font-size:16px;outline:none;transition:border-color .2s}
  input[type=text]:focus{border-color:#4F46E5;box-shadow:0 0 0 4px rgba(79,70,229,.12)}
  button{padding:13px 22px;border:none;border-radius:12px;font-size:14px;font-weight:600;
         cursor:pointer;transition:opacity .2s,transform .1s}
  button:active{transform:scale(.97)} button:hover{opacity:.9}
  .btn-primary{background:#4F46E5;color:white;box-shadow:0 4px 12px rgba(79,70,229,.3)}
  .btn-teal{background:#0F766E;color:white;box-shadow:0 4px 12px rgba(15,118,110,.2)}
  .btn-sec{background:#E2E8F0;color:#475569}
  .result{display:none;margin-top:20px;padding:22px 24px;border-radius:14px;
          border:2px solid #EEF2FF;background:#FAFBFF}
  .result.visible{display:block}
  .result-row{display:flex;align-items:center;gap:20px;margin-bottom:18px;flex-wrap:wrap}
  .r-label{font-size:11px;color:#94A3B8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
  .r-input{font-size:18px;color:#475569}
  .r-arrow{font-size:26px;color:#CBD5E1}
  .r-pred{font-size:28px;font-weight:800;color:#4F46E5}
  .conf-lbl{font-size:12px;color:#64748B;margin-bottom:6px;font-weight:500}
  .bar{background:#E2E8F0;border-radius:8px;height:12px;overflow:hidden}
  .bar-fill{height:100%;border-radius:8px;transition:width .6s}
  .tc{padding:14px 16px;border-left:4px solid #4F46E5;background:#0F172A;color:#E2E8F0;
      margin-bottom:10px;border-radius:0 8px 8px 0;font-family:'SF Mono',Consolas,monospace}
  .tc-title{font-weight:700;color:#34D399;font-size:13px;margin-bottom:5px}
  .tc-line{font-size:13px;line-height:1.75}
  .tc-line b{color:#F0F9FF}
  .ok{color:#34D399;font-weight:700;margin-left:6px}
  .no{color:#F87171;font-weight:700;margin-left:6px}
  .alts{margin-top:16px;border-top:1px solid #E2E8F0;padding-top:16px}
  .alt-title{font-size:12px;color:#64748B;text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px;font-weight:600}
  .alt-item{display:flex;align-items:center;justify-content:space-between;background:white;padding:12px 16px;border-radius:10px;margin-bottom:8px;box-shadow:0 1px 3px rgba(0,0,0,.05);border:1px solid #F1F5F9}
  .alt-name{font-weight:600;color:#334155;font-size:15px}
  .alt-conf-wrap{display:flex;align-items:center;gap:10px;flex:1;max-width:200px;justify-content:flex-end}
  .alt-conf-lbl{font-size:12px;color:#64748B;font-weight:500;min-width:45px;text-align:right}
  .alt-bar{background:#E2E8F0;border-radius:6px;height:6px;width:80px;overflow:hidden}
  .alt-bar-fill{height:100%;border-radius:6px;transition:width .6s}
</style></head><body>

<div class="hero">
  <h1>💊 Medicine Name Reconstruction</h1>
  <p>LSTM + Dataset Snap · Corrects misspelled medicine names</p>
</div>

<div class="card">
  <h2>Try It</h2>
  <div class="row">
    <input type="text" id="inp" placeholder="e.g. panadl, amlodipne, morhine..." autofocus>
    <button class="btn-primary" onclick="predict()">🔍 Reconstruct</button>
    <button class="btn-sec" onclick="resetAll()">↺</button>
  </div>
  <div class="result" id="result">
    <div class="result-row">
      <div><div class="r-label">Input</div><div class="r-input" id="r-input"></div></div>
      <div class="r-arrow">→</div>
      <div><div class="r-label">Predicted</div><div class="r-pred" id="r-pred"></div></div>
    </div>
    <div class="conf-lbl" id="r-conf"></div>
    <div class="bar"><div class="bar-fill" id="r-bar" style="width:0%"></div></div>
    <div id="alternatives"></div>
  </div>
</div>

<div class="card">
  <h2>Test Cases (Reference Image Format)</h2>
  <button class="btn-teal" onclick="runBatch()">📋 Run Test Cases</button>
  <div id="tests" style="margin-top:14px"></div>
</div>

<script>
const DEMOS=[
  {input:'panadl',expected:'panadol'},
  {input:'amlodipne',expected:'amlodipine'},
  {input:'hydrcodone',expected:'hydrocodone'},
  {input:'morhine',expected:'morphine'},
  {input:'paracetmol',expected:'paracetamol'},
  {input:'asprin',expected:'aspirin'},
  {input:'ibuprofn',expected:'ibuprofen'}
];

function confColor(c){return c>=.85?'#10B981':c>=.60?'#F59E0B':'#EF4444'}
function confLabel(c){return (c>=.85?'🟢 High ':c>=.60?'🟡 Medium ':'🔴 Low ')+(c*100).toFixed(1)+'%'}

async function predict(){
  const val=document.getElementById('inp').value.trim();
  if(!val) return;
  const res=await fetch('/predict',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({input:val})});
  const data=await res.json();
  document.getElementById('r-input').textContent=val;
  document.getElementById('r-pred').textContent=data.prediction;
  document.getElementById('r-conf').textContent='Confidence · '+confLabel(data.confidence);
  const b=document.getElementById('r-bar');
  b.style.background=confColor(data.confidence);
  b.style.width=(data.confidence*100)+'%';
  
  const alts = data.alternatives || [];
  const altsDiv = document.getElementById('alternatives');
  if(alts.length > 0){
    altsDiv.innerHTML = '<div class="alts"><div class="alt-title">Other Matches</div>' + 
      alts.map(a => `
        <div class="alt-item">
          <div class="alt-name">${a.prediction}</div>
          <div class="alt-conf-wrap">
            <div class="alt-bar"><div class="alt-bar-fill" style="width:${a.confidence*100}%;background:${confColor(a.confidence)}"></div></div>
            <div class="alt-conf-lbl">${(a.confidence*100).toFixed(1)}%</div>
          </div>
        </div>
      `).join('') + '</div>';
  } else {
    altsDiv.innerHTML = '';
  }
  
  document.getElementById('result').classList.add('visible');
}

async function runBatch(){
  const host=document.getElementById('tests');
  host.innerHTML='<p style="color:#94A3B8">Running...</p>';
  const res=await fetch('/predict_batch',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({inputs:DEMOS.map(d=>d.input)})});
  const data=await res.json();
  host.innerHTML=data.predictions.map((p,i)=>{
    const exp=DEMOS[i].expected,ok=p.prediction===exp;
    return `<div class="tc">
      <div class="tc-title">Test Case ${i+1}:</div>
      <div class="tc-line">Input: <b>${p.input}</b></div>
      <div class="tc-line">Predicted Output: <b>${p.prediction}</b></div>
      <div class="tc-line">Expected Output: <b>${exp}</b>${ok?'<span class="ok">✓</span>':'<span class="no">✗</span>'}</div>
    </div>`;
  }).join('');
}

function resetAll(){
  document.getElementById('inp').value='';
  document.getElementById('result').classList.remove('visible');
  document.getElementById('tests').innerHTML='';
}
document.getElementById('inp').addEventListener('keydown',e=>{if(e.key==='Enter')predict()});
</script></body></html>
"""


@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get('input', '')).strip()
    if not text: return jsonify({'error': 'empty input'}), 400
    result = predict_one(text); result['input'] = text
    return jsonify(result)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json(silent=True) or {}
    inputs = data.get('inputs', [])
    if not isinstance(inputs, list) or not inputs:
        return jsonify({'error': 'inputs required'}), 400
    results = []
    for inp in inputs:
        inp = str(inp).strip()
        if inp:
            r = predict_one(inp); r['input'] = inp
            results.append(r)
    return jsonify({'predictions': results})


if __name__ == '__main__':
    load_artifacts()
    port = int(os.environ.get('PORT', 5000))
    log.info("🚀 http://localhost:%d", port)
    app.run(host='0.0.0.0', port=port, debug=False)
