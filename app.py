from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # Créer le dossier static s'il n'existe pas
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
