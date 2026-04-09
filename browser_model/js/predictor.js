/**
 * browser_model/js/predictor.js
 * Engine for loading ONNX models in browser using WASM
 */

class BehavioralPredictor {
    constructor() {
        this.session = null;
        this.vocab = {};
        this.isReady = false;
        this.targetSeqLen = 20; // Must match training config
    }

    async initialize() {
        try {
            // In a real deployed app, the path points to backend /models/download API
            // For the demo without backend running, we expect the model file in models/ directory
            // We use ort.js from CDN
            
            // Allow WASM threads if environment supports it
            ort.env.wasm.numThreads = 1;
            
            this.updateStatus('Loading vocabularies...', 'warning');
            await this.loadVocabularies();
            
            this.updateStatus('Loading ONNX Model (<1MB)...', 'warning');
            
            // Note: Since this is meant to be browser deployable, users serve this folder
            const modelPath = './behavioral_predictor.onnx';
            
            try {
                this.session = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'] });
            } catch (e) {
                console.warn("Could not load local model, starting in mock mode for UI demo. Real deployment requires actual .onnx file.", e);
                this.updateStatus('UI Demo Mode (Model Missing)', 'success');
                this.isReady = true;
                this.mockMode = true;
                return;
            }

            this.isReady = true;
            this.updateStatus('Model Ready', 'success');
        } catch (error) {
            console.error('Initialization error:', error);
            this.updateStatus('Error loading model', 'error');
        }
    }

    async loadVocabularies() {
        // Fallback mock vocab if network restricted / local file testing
        this.vocab = {
            event_name: { "<PAD>":0, "scroll":1, "add_to_cart":2, "viewcontent":3, "pageview":4, "search":5, "purchase":6 },
            device_os: { "<PAD>":0, "android":1, "ios":2, "desktop":3 },
            channel: { "<PAD>":0, "browser":1, "app":2 },
            category: { "<PAD>":0, "books":1, "beauty":2, "electronics":3, "fashion":4, "sports":5, "home":6, "toys":7, "grocery":8, "auto":9, "health":10, "music":11, "games":12, "other":13 },
            traffic_source: { "<PAD>":0, "direct":1, "organic":2, "paid":3, "referral":4, "social":5 }
        };
        
        // Attempt to load from JSON, silently fallback to generated
        try {
            const res = await fetch('./vocab/event_name.json');
            if (res.ok) this.vocab.event_name = await res.json();
            
            const res2 = await fetch('./vocab/device_os.json');
            if (res2.ok) this.vocab.device_os = await res2.json();
            
            const res3 = await fetch('./vocab/channel.json');
            if (res3.ok) this.vocab.channel = await res3.json();
        } catch (e) {
            console.log("Using default vocab map");
        }
    }

    encodeVocab(category, value) {
        if (!this.vocab[category]) return 1; // UNK
        return this.vocab[category][value] !== undefined ? this.vocab[category][value] : 1;
    }

    async predict(events) {
        if (!this.isReady) return null;
        
        const startTime = performance.now();
        
        // Take last N events
        const recentEvents = events.slice(-this.targetSeqLen);
        const padLen = this.targetSeqLen - recentEvents.length;
        
        // Prepare categorical tensors [1, seq_len]
        const event_ids = new BigInt64Array(this.targetSeqLen);
        const device_ids = new BigInt64Array(this.targetSeqLen);
        const channel_ids = new BigInt64Array(this.targetSeqLen);
        const category_ids = new BigInt64Array(this.targetSeqLen);
        const hour_ids = new BigInt64Array(this.targetSeqLen);
        const traffic_ids = new BigInt64Array(this.targetSeqLen);
        const padding_mask = new Uint8Array(this.targetSeqLen);
        
        // Prepare numeric features [1, 6]
        const numeric_features = new Float32Array(6);
        
        const currentHour = new Date().getHours();
        
        // Pad prefix
        for (let i = 0; i < padLen; i++) {
            event_ids[i] = 0n;
            device_ids[i] = 0n;
            channel_ids[i] = 0n;
            category_ids[i] = 0n;
            hour_ids[i] = 0n;
            traffic_ids[i] = 0n;
            padding_mask[i] = 1; // true = padding
        }
        
        // Fill actual
        recentEvents.forEach((ev, i) => {
            const idx = padLen + i;
            event_ids[idx] = BigInt(this.encodeVocab('event_name', ev.event_name));
            device_ids[idx] = BigInt(this.encodeVocab('device_os', 'desktop'));
            channel_ids[idx] = BigInt(this.encodeVocab('channel', 'browser'));
            category_ids[idx] = 0n; // Use vocab map if needed
            hour_ids[idx] = BigInt(currentHour);
            traffic_ids[idx] = BigInt(this.encodeVocab('traffic_source', 'direct'));
            padding_mask[idx] = 0; // false
        });

        // Compute 6 numeric statistics for fusion
        if (recentEvents.length > 0) {
            const num = recentEvents.length;
            numeric_features[0] = recentEvents.filter(e => e.event_name === 'purchase').length / num;
            numeric_features[1] = recentEvents.filter(e => e.event_name === 'add_to_cart').length / num;
            numeric_features[2] = recentEvents.filter(e => e.event_name === 'scroll').length / num;
            numeric_features[3] = new Set(recentEvents.map(e => e.event_name)).size / 10;
            numeric_features[4] = 0.5; // avg_time_delta mock
            numeric_features[5] = 1.0; // session_count mock
        }

        let results;
        if (this.mockMode) {
            results = this.generateMockPredictions(recentEvents);
            // Artificial delay to simulate wasm processing
            await new Promise(r => setTimeout(r, 5)); 
        } else {
            const feeds = {
                "event_ids": new ort.Tensor('int64', event_ids, [1, this.targetSeqLen]),
                "device_ids": new ort.Tensor('int64', device_ids, [1, this.targetSeqLen]),
                "channel_ids": new ort.Tensor('int64', channel_ids, [1, this.targetSeqLen]),
                "category_ids": new ort.Tensor('int64', category_ids, [1, this.targetSeqLen]),
                "hour_ids": new ort.Tensor('int64', hour_ids, [1, this.targetSeqLen]),
                "traffic_ids": new ort.Tensor('int64', traffic_ids, [1, this.targetSeqLen]),
                "numeric_features": new ort.Tensor('float32', numeric_features, [1, 6]),
                "padding_mask": new ort.Tensor('bool', padding_mask, [1, this.targetSeqLen])
            };

            const output = await this.session.run(feeds);
            results = this.processOutputs(output);
        }

        const endTime = performance.now();
        results.inferenceTimeMs = (endTime - startTime).toFixed(1);
        
        return results;
    }
    
    processOutputs(outputs) {
        // Translate tensor outputs to JSON predictions
        // Outputs keys typically match exporter output_names
        
        const sigmoid = x => 1 / (1 + Math.exp(-x));
        const argmax = arr => arr.indexOf(Math.max(...arr));
        
        // Accessing underlying data array
        const p_pur = sigmoid(outputs["p_purchase"].data[0]);
        const p_churn = sigmoid(outputs["p_churn"].data[0]);
        
        const next_event_idx = argmax(Array.from(outputs["p_next_event"].data));
        const channel_idx = argmax(Array.from(outputs["p_channel"].data));
        
        // Reverse mapping vocab
        let next_event = "unknown";
        for (const [k, v] of Object.entries(this.vocab.event_name)) {
            if (v === next_event_idx) next_event = k;
        }
        
        return {
            purchase_probability: (p_pur * 100).toFixed(1),
            churn_risk: (p_churn * 100).toFixed(1),
            next_event: next_event,
            preferred_channel: channel_idx === 2 ? 'browser' : 'app',
            engagement_score: outputs["p_engagement"].data[0].toFixed(2),
            inactivity_risk: Math.max(0, outputs["p_inactivity"].data[0]).toFixed(1),
            active_time: "Afternoon" // simplified enum decode
        };
    }

    generateMockPredictions(events) {
        // A simple heuristic mock just to make the UI look responsive if model file is absent 
        // Emulates model reacting to sequences
        const hasCart = events.some(e => e.event_name === 'add_to_cart');
        const count = events.length;
        
        let purProb = Math.min(95, count * 5 + (hasCart ? 40 : 0));
        let churnRisk = Math.max(5, 80 - count * 10);
        let nextEvent = hasCart ? "purchase" : (count > 2 ? "add_to_cart" : "viewcontent");
        
        if (events.length > 0 && events[events.length - 1].event_name === 'purchase') {
            purProb = 2.0;
            churnRisk = 60.0;
            nextEvent = "pageview";
        }

        return {
            purchase_probability: purProb.toFixed(1),
            churn_risk: churnRisk.toFixed(1),
            next_event: nextEvent,
            preferred_channel: "browser",
            engagement_score: (count * 0.15).toFixed(2),
            inactivity_risk: (10 - count).toFixed(1),
            active_time: "Evening"
        };
    }

    updateStatus(message, state) {
        const text = document.getElementById('model-status-text');
        const dot = document.getElementById('model-status-dot');
        if (text && dot) {
            text.textContent = message;
            dot.className = `dot ${state}`;
        }
    }
}

// Global instance
window.predictor = new BehavioralPredictor();
// Start init on page load
window.addEventListener('DOMContentLoaded', () => {
    window.predictor.initialize();
});
