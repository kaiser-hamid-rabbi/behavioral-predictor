/**
 * browser_model/js/event_simulator.js
 * Manages the UI buffers and triggers predictions
 */

class EventSimulator {
    constructor() {
        this.eventBuffer = [];
        this.maxBuffer = 20;
        
        this.eventListEl = document.getElementById('event-list');
        this.countEl = document.getElementById('event-count');
        this.timeEl = document.getElementById('inference-time');
        
        this.setupListeners();
    }

    setupListeners() {
        const buttons = document.querySelectorAll('.btn-action');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                this.addEvent(
                    btn.getAttribute('data-event'), 
                    btn.getAttribute('data-cat')
                );
            });
        });
    }

    async addEvent(eventName, category) {
        // Maintain buffer size
        if (this.eventBuffer.length >= this.maxBuffer) {
            this.eventBuffer.shift();
        }

        const newEvent = {
            event_name: eventName,
            category: category,
            timestamp: new Date().toISOString()
        };

        this.eventBuffer.push(newEvent);
        
        this.updateUI();
        await this.runPrediction();
    }

    updateUI() {
        // Clear empty state
        if (this.eventBuffer.length === 1 && this.eventListEl.querySelector('.empty-state')) {
            this.eventListEl.innerHTML = '';
        }

        // Render buffer
        this.eventListEl.innerHTML = '';
        
        // Render in reverse (newest on top)
        const reversed = [...this.eventBuffer].reverse();
        
        reversed.forEach(ev => {
            const div = document.createElement('div');
            div.className = 'event-item';
            div.innerHTML = `
                <span class="event-name">${this.formatName(ev.event_name)}</span>
                <span class="event-cat">${ev.category}</span>
            `;
            this.eventListEl.appendChild(div);
        });

        this.countEl.textContent = `${this.eventBuffer.length}/${this.maxBuffer}`;
    }

    formatName(name) {
        return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    }

    async runPrediction() {
        if (!window.predictor || !window.predictor.isReady) return;

        const results = await window.predictor.predict(this.eventBuffer);
        
        if (results) {
            this.updatePredictionsUI(results);
        }
    }

    updatePredictionsUI(res) {
        // Update values
        document.getElementById('pred-purchase').textContent = res.purchase_probability;
        document.getElementById('pred-churn').textContent = res.churn_risk;
        document.getElementById('pred-next-event').textContent = this.formatName(res.next_event);
        document.getElementById('pred-channel').textContent = this.formatName(res.preferred_channel);
        document.getElementById('pred-engagement').textContent = res.engagement_score;
        document.getElementById('pred-inactivity').textContent = res.inactivity_risk;
        document.getElementById('pred-time').textContent = res.active_time;
        
        // Update bars
        const pb = document.getElementById('bar-purchase');
        pb.style.width = `${res.purchase_probability}%`;
        
        // Color transition logic for purchase (green/red)
        if (parseFloat(res.purchase_probability) > 50) {
            pb.className = 'progress-bar fill-success';
        } else {
            pb.className = 'progress-bar fill-warning';
        }

        const cb = document.getElementById('bar-churn');
        cb.style.width = `${res.churn_risk}%`;
        if (parseFloat(res.churn_risk) > 50) {
            cb.className = 'progress-bar fill-danger';
        } else {
            cb.className = 'progress-bar fill-success'; // Low churn is good
        }

        // Update timing
        this.timeEl.textContent = `${res.inferenceTimeMs}ms`;
    }
}

// Initialize simulator when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    window.simulator = new EventSimulator();
});
