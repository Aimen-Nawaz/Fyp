"""
Model loading utility for medicine reconstruction app
Supports both local and remote (cloud storage) model loading
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import tempfile
import shutil

logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {}


def get_artifacts_path() -> str:
    """
    Get the path to model artifacts.
    With Git LFS: ./model_artifacts/ is tracked in git
    Vercel automatically deploys with the app
    """
    # Model artifacts folder (tracked with Git LFS)
    local_path = Path(__file__).parent.parent / "model_artifacts"
    
    if not local_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found at {local_path}. "
            f"Ensure model_artifacts/ is tracked with Git LFS and deployed with app."
        )
    
    return str(local_path)


def check_model_files() -> Dict[str, bool]:
    """Check which model files exist"""
    artifacts_path = get_artifacts_path()
    files = {
        'config': 'model_config.json',
        't2i': 'token_to_idx.json',
        'i2t': 'idx_to_token.json',
        'model': 'medicine_lstm.keras',
        'names': 'known_names.txt',
    }
    
    status = {}
    for key, filename in files.items():
        filepath = Path(artifacts_path) / filename
        status[key] = filepath.exists()
    
    return status


def load_artifacts():
    """
    Load model artifacts from local file system.
    Model artifacts are tracked with Git LFS and deployed with app to Vercel.
    """
    global _model_cache
    
    if _model_cache:
        logger.info("✓ Model already loaded from cache")
        return _model_cache
    
    artifacts_path = get_artifacts_path()
    
    paths = {
        'config': Path(artifacts_path) / 'model_config.json',
        't2i': Path(artifacts_path) / 'token_to_idx.json',
        'i2t': Path(artifacts_path) / 'idx_to_token.json',
        'model': Path(artifacts_path) / 'medicine_lstm.keras',
        'names': Path(artifacts_path) / 'known_names.txt',
    }
    
    # Check for missing files
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        logger.error(f"❌ Missing model files: {missing}")
        logger.error(f"   Expected in: {artifacts_path}")
        raise FileNotFoundError(
            f"Model files not found: {missing}. "
            f"Ensure model_artifacts/ folder is tracked with Git LFS and deployed with the app."
        )
    
    try:
        # Load configuration
        with open(paths['config']) as f:
            config = json.load(f)
        logger.info(f"✓ Loaded config from {paths['config']}")
        
        # Load tokenizers
        with open(paths['t2i']) as f:
            token_to_idx = json.load(f)
        logger.info(f"✓ Loaded token_to_idx ({len(token_to_idx)} tokens)")
        
        with open(paths['i2t']) as f:
            idx_to_token = {int(k): v for k, v in json.load(f).items()}
        logger.info(f"✓ Loaded idx_to_token ({len(idx_to_token)} tokens)")
        
        # Load known medicine names
        with open(paths['names'], encoding='utf-8') as f:
            known_names = sorted(set(line.strip().lower() for line in f if line.strip()))
        logger.info(f"✓ Loaded {len(known_names)} known medicine names")
        
        # Load model
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow.keras.optimizers import Adam
        
        PAD_IDX = config['pad_idx']
        
        @tf.function
        def masked_sparse_ce(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            mask = tf.cast(tf.not_equal(y_true, PAD_IDX), tf.float32)
            return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
        
        class MaskedAccuracy(tf.keras.metrics.Metric):
            def __init__(self, pad_idx=PAD_IDX, name='masked_accuracy', **kw):
                super().__init__(name=name, **kw)
                self.pad_idx = pad_idx
                self.correct = self.add_weight(name='correct', initializer='zeros')
                self.total = self.add_weight(name='total', initializer='zeros')
            
            def update_state(self, yt, yp, sw=None):
                yt = tf.cast(tf.reshape(yt, [-1]), tf.int32)
                yp = tf.cast(tf.argmax(yp, axis=-1), tf.int32)
                yp = tf.reshape(yp, [-1])
                m = tf.not_equal(yt, self.pad_idx)
                self.correct.assign_add(tf.reduce_sum(tf.cast(
                    tf.equal(tf.boolean_mask(yt, m), tf.boolean_mask(yp, m)), tf.float32)))
                self.total.assign_add(tf.cast(tf.reduce_sum(tf.cast(m, tf.int32)), tf.float32))
            
            def result(self):
                return self.correct / (self.total + 1e-8)
            
            def reset_state(self):
                self.correct.assign(0.)
                self.total.assign(0.)
        
        model = tf.keras.models.load_model(
            str(paths['model']),
            custom_objects={'masked_sparse_ce': masked_sparse_ce, 'MaskedAccuracy': MaskedAccuracy},
            compile=False,
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss=masked_sparse_ce,
                      metrics=[MaskedAccuracy(PAD_IDX)])
        logger.info(f"✓ Model loaded successfully")
        
        # Cache results
        _model_cache = {
            'model': model,
            'token_to_idx': token_to_idx,
            'idx_to_token': idx_to_token,
            'config': config,
            'known_names': known_names,
        }
        
        return _model_cache
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}", exc_info=True)
        raise


def get_cached_model():
    """Get cached model (load if not already loaded)"""
    if not _model_cache:
        load_artifacts()
    return _model_cache
