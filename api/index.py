"""
Vercel Serverless API Handler for Medicine Name Reconstruction
Main entry point for all HTTP requests
"""
import os
import json
import logging
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import prediction modules
from model_utils import get_cached_model, check_model_files
from predict import predict_one, predict_batch


# HTML UI Template
HTML_TEMPLATE = """<!DOCTYPE html>
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
  <p>BiLSTM + Dataset Snap · Corrects misspelled medicine names</p>
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
  {input:'asati',expected:'avastin'},
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
  const res=await fetch('/api/predict',{method:'POST',
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
          <div class="alt-name">\${a.prediction}</div>
          <div class="alt-conf-wrap">
            <div class="alt-bar"><div class="alt-bar-fill" style="width:\${a.confidence*100}%;background:\${confColor(a.confidence)}"></div></div>
            <div class="alt-conf-lbl">\${(a.confidence*100).toFixed(1)}%</div>
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
  const res=await fetch('/api/predict_batch',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({inputs:DEMOS.map(d=>d.input)})});
  const data=await res.json();
  host.innerHTML=data.predictions.map((p,i)=>{
    const exp=DEMOS[i].expected,ok=p.prediction===exp;
    return `<div class="tc">
      <div class="tc-title">Test Case \${i+1}:</div>
      <div class="tc-line">Input: <b>\${p.input}</b></div>
      <div class="tc-line">Predicted Output: <b>\${p.prediction}</b></div>
      <div class="tc-line">Expected Output: <b>\${exp}</b>\${ok?'<span class="ok">✓</span>':'<span class="no">✗</span>'}</div>
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


class MedicineHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for medicine name prediction"""
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(format % args)
    
    def _send_json(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_html(self, html: str, status_code: int = 200):
        """Send HTML response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path
        
        if path == '/' or path == '':
            self._send_html(HTML_TEMPLATE)
        elif path == '/health':
            self._send_json({'status': 'ok', 'model_files': check_model_files()})
        else:
            self._send_json({'error': f'Not found: {path}'}, 404)
    
    def do_POST(self):
        """Handle POST requests"""
        path = urlparse(self.path).path
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body) if body else {}
        except Exception as e:
            logger.error(f"Error parsing request: {e}")
            self._send_json({'error': 'Invalid request'}, 400)
            return
        
        try:
            if path == '/api/predict' or path == '/predict':
                self._handle_predict(data)
            elif path == '/api/predict_batch' or path == '/predict_batch':
                self._handle_predict_batch(data)
            else:
                self._send_json({'error': f'Not found: {path}'}, 404)
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            self._send_json({'error': str(e)}, 500)
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _handle_predict(self, data: Dict):
        """Handle single prediction"""
        text = str(data.get('input', '')).strip()
        if not text:
            self._send_json({'error': 'empty input'}, 400)
            return
        
        try:
            model_artifacts = get_cached_model()
            result = predict_one(text, model_artifacts)
            result['input'] = text
            self._send_json(result)
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            self._send_json({'error': f'Prediction failed: {str(e)}'}, 500)
    
    def _handle_predict_batch(self, data: Dict):
        """Handle batch prediction"""
        inputs = data.get('inputs', [])
        if not isinstance(inputs, list) or not inputs:
            self._send_json({'error': 'inputs required'}, 400)
            return
        
        try:
            model_artifacts = get_cached_model()
            predictions = predict_batch(inputs, model_artifacts)
            self._send_json({'predictions': predictions})
        except Exception as e:
            logger.error(f"Batch prediction error: {e}", exc_info=True)
            self._send_json({'error': f'Batch prediction failed: {str(e)}'}, 500)


# WSGI application for Vercel
def handler(environ, start_response):
    """WSGI handler for Vercel serverless"""
    # Parse request
    method = environ['REQUEST_METHOD']
    path = environ['PATH_INFO']
    
    logger.info(f"{method} {path}")
    
    try:
        if method == 'GET':
            if path == '/' or path == '':
                start_response('200 OK', [('Content-Type', 'text/html; charset=utf-8')])
                return [HTML_TEMPLATE.encode()]
            elif path == '/health':
                start_response('200 OK', [('Content-Type', 'application/json')])
                return [json.dumps({
                    'status': 'ok',
                    'model_files': check_model_files()
                }).encode()]
            else:
                start_response('404 Not Found', [('Content-Type', 'application/json')])
                return [json.dumps({'error': f'Not found: {path}'}).encode()]
        
        elif method == 'POST':
            # Read request body
            try:
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                body = environ['wsgi.input'].read(content_length).decode('utf-8')
                data = json.loads(body) if body else {}
            except Exception as e:
                logger.error(f"Error parsing request: {e}")
                start_response('400 Bad Request', [('Content-Type', 'application/json')])
                return [json.dumps({'error': 'Invalid request'}).encode()]
            
            if path == '/api/predict' or path == '/predict':
                text = str(data.get('input', '')).strip()
                if not text:
                    start_response('400 Bad Request', [('Content-Type', 'application/json')])
                    return [json.dumps({'error': 'empty input'}).encode()]
                
                try:
                    model_artifacts = get_cached_model()
                    result = predict_one(text, model_artifacts)
                    result['input'] = text
                    start_response('200 OK', [('Content-Type', 'application/json')])
                    return [json.dumps(result).encode()]
                except Exception as e:
                    logger.error(f"Prediction error: {e}", exc_info=True)
                    start_response('500 Internal Server Error', [('Content-Type', 'application/json')])
                    return [json.dumps({'error': f'Prediction failed: {str(e)}'}).encode()]
            
            elif path == '/api/predict_batch' or path == '/predict_batch':
                inputs = data.get('inputs', [])
                if not isinstance(inputs, list) or not inputs:
                    start_response('400 Bad Request', [('Content-Type', 'application/json')])
                    return [json.dumps({'error': 'inputs required'}).encode()]
                
                try:
                    model_artifacts = get_cached_model()
                    predictions = predict_batch(inputs, model_artifacts)
                    start_response('200 OK', [('Content-Type', 'application/json')])
                    return [json.dumps({'predictions': predictions}).encode()]
                except Exception as e:
                    logger.error(f"Batch prediction error: {e}", exc_info=True)
                    start_response('500 Internal Server Error', [('Content-Type', 'application/json')])
                    return [json.dumps({'error': f'Batch prediction failed: {str(e)}'}).encode()]
            else:
                start_response('404 Not Found', [('Content-Type', 'application/json')])
                return [json.dumps({'error': f'Not found: {path}'}).encode()]
        
        elif method == 'OPTIONS':
            start_response('200 OK', [
                ('Access-Control-Allow-Origin', '*'),
                ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
                ('Access-Control-Allow-Headers', 'Content-Type'),
            ])
            return [b'']
        
        else:
            start_response('405 Method Not Allowed', [('Content-Type', 'application/json')])
            return [json.dumps({'error': f'Method not allowed: {method}'}).encode()]
    
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        start_response('500 Internal Server Error', [('Content-Type', 'application/json')])
        return [json.dumps({'error': 'Internal server error'}).encode()]
