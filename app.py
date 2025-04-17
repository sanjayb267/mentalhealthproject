# app.py
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import plotly.graph_objs as go
import plotly
import json
import torch
import numpy as np
import datetime
import random

app = Flask(__name__)
CORS(app)
app.secret_key = 'super-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'

# ~~~~~~~~~~~ AI Services ~~~~~~~~~~~
class AICore:
    def __init__(self):
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        self.therapy_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.therapy_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.coping_strategies = {
            'anger': ['Deep breathing', 'Physical exercise', 'Progressive muscle relaxation'],
            'sadness': ['Gratitude journaling', 'Social connection', 'Creative expression'],
            'anxiety': ['Grounding techniques', 'Mindful observation', 'Thought challenging']
        }

    def analyze_emotion(self, text):
        results = self.emotion_analyzer(text)[0]
        return {result['label'].lower(): result['score'] for result in results}

    def generate_response(self, text):
        inputs = self.therapy_tokenizer.encode(text + self.therapy_tokenizer.eos_token, return_tensors="pt")
        outputs = self.therapy_model.generate(
            inputs,
            max_length=1000,
            temperature=0.9,
            top_k=150,
            repetition_penalty=1.2
        )
        return self.therapy_tokenizer.decode(outputs[0], skip_special_tokens=True)

ai_core = AICore()

# ~~~~~~~~~~~ Interactive Features ~~~~~~~~~~~
class MindSpace:
    def __init__(self):
        self.history = []
        self.mood_journal = []
        self.active_visualization = None

    def add_entry(self, entry):
        self.history.append({
            **entry,
            'timestamp': datetime.datetime.now().isoformat(),
            'id': len(self.history)+1
        })
        if len(self.history) > 50:
            self.history.pop(0)

mind_space = MindSpace()

# ~~~~~~~~~~~ Routes ~~~~~~~~~~~
@app.route('/')
def cosmic_dashboard():
    return render_template('cosmic_dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze_universe():
    data = request.get_json()
    try:
        # Emotional Analysis
        emotions = ai_core.analyze_emotion(data['text'])
        dominant_emotion = max(emotions, key=emotions.get)
        
        # AI Response
        ai_response = ai_core.generate_response(data['text'])
        
        # Coping Strategies
        strategies = ai_core.coping_strategies.get(dominant_emotion, ['Mindful breathing'])
        
        # 3D Visualization Data
        visualization = create_galaxy_view(emotions)
        
        # Store session
        entry = {
            'text': data['text'],
            'emotion': dominant_emotion,
            'intensity': emotions[dominant_emotion],
            'response': ai_response,
            'strategies': strategies
        }
        mind_space.add_entry(entry)
        
        return jsonify({
            'response': ai_response,
            'emotion': dominant_emotion,
            'intensity': emotions[dominant_emotion],
            'strategies': strategies,
            'visualization': visualization,
            'history': mind_space.history[-5:]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_galaxy_view(emotions):
    traces = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
    for i, (emotion, score) in enumerate(emotions.items()):
        traces.append({
            'x': [random.gauss(0, 1) for _ in range(100)],
            'y': [random.gauss(0, 1) for _ in range(100)],
            'z': [random.gauss(0, 1) for _ in range(100)],
            'mode': 'markers',
            'marker': {
                'size': score*20,
                'color': colors[i % len(colors)],
                'opacity': 0.7
            },
            'name': emotion.capitalize()
        })
    return traces

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)