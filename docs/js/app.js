document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    const predictionElement = document.getElementById('prediction');
    const confidenceBars = document.querySelector('.confidence-bars');
    
    let isDrawing = false;
    let model = null;
    
    // Initialisation du canvas
    function initCanvas() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'white';
        
        // Événements de dessin
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Pour le support tactile
        canvas.addEventListener('touchstart', handleTouch);
        canvas.addEventListener('touchmove', handleTouch);
        canvas.addEventListener('touchend', stopDrawing);
        
        // Boutons
        predictBtn.addEventListener('click', predict);
        clearBtn.addEventListener('click', clearCanvas);
    }
    
    // Gestion du toucher
    function handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(
            e.type === 'touchstart' ? 'mousedown' : 'mousemove',
            {
                clientX: touch.clientX,
                clientY: touch.clientY
            }
        );
        if (e.type === 'touchstart') {
            startDrawing(mouseEvent);
        } else {
            draw(mouseEvent);
        }
    }
    
    function startDrawing(e) {
        isDrawing = true;
        draw(e);
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }
    
    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath();
    }
    
    function clearCanvas() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        predictionElement.textContent = '?';
        updateConfidenceBars([]);
    }
    
    // Chargement du modèle ONNX
    async function loadModel() {
        try {
            model = await ort.InferenceSession.create('../mnist_model.onnx');
            console.log('Modèle chargé avec succès!');
        } catch (error) {
            console.error('Erreur lors du chargement du modèle:', error);
            alert('Erreur lors du chargement du modèle. Vérifiez la console pour plus de détails.');
        }
    }
    
    // Prédiction
    async function predict() {
        if (!model) {
            alert('Le modèle n\'est pas encore chargé. Veuillez patienter...');
            return;
        }
        
        try {
            // Préparation de l'image
            const imageData = preprocessImage();
            
            // Création des tenseurs d'entrée
            const inputTensor = new ort.Tensor('float32', new Float32Array(imageData), [1, 1, 28, 28]);
            
            // Exécution du modèle
            const outputMap = await model.run({ input: inputTensor });
            const output = outputMap.output.data;
            
            // Récupération de la prédiction
            const predictedClass = output.indexOf(Math.max(...output));
            
            // Conversion des log-probabilités en probabilités
            const probabilities = [];
            for (let i = 0; i < output.length; i++) {
                probabilities.push(Math.exp(output[i]));
            }
            
            // Mise à jour de l'interface
            predictionElement.textContent = predictedClass;
            updateConfidenceBars(probabilities);
            
        } catch (error) {
            console.error('Erreur lors de la prédiction:', error);
            alert('Erreur lors de la prédiction. Vérifiez la console pour plus de détails.');
        }
    }
    
    // Prétraitement de l'image
    function preprocessImage() {
        // Création d'un canvas temporaire pour le redimensionnement
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        
        // Redimensionnement de l'image à 28x28 pixels
        tempCtx.drawImage(canvas, 0, 0, 28, 28);
        
        // Récupération des données de l'image
        const imgData = tempCtx.getImageData(0, 0, 28, 28);
        const data = imgData.data;
        
        // Conversion en niveaux de gris et normalisation
        const processedData = [];
        for (let i = 0; i < data.length; i += 4) {
            // Conversion en niveaux de gris (moyenne des canaux RGB)
            const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
            // Normalisation entre -0.5 et 0.5 (comme dans l'entraînement)
            const normalized = (gray / 255 - 0.1307) / 0.3081;
            processedData.push(normalized);
        }
        
        return processedData;
    }
    
    // Mise à jour des barres de confiance
    function updateConfidenceBars(probabilities) {
        // Effacer les barres existantes
        confidenceBars.innerHTML = '';
        
        if (probabilities.length === 0) return;
        
        // Créer une barre pour chaque chiffre (0-9)
        for (let i = 0; i < 10; i++) {
            const probability = probabilities[i] || 0;
            const height = (probability * 100).toFixed(1);
            
            const barContainer = document.createElement('div');
            barContainer.style.display = 'flex';
            barContainer.style.flexDirection = 'column';
            barContainer.style.alignItems = 'center';
            
            const valueLabel = document.createElement('div');
            valueLabel.className = 'confidence-value';
            valueLabel.textContent = `${height}%`;
            
            const bar = document.createElement('div');
            bar.className = 'confidence-bar';
            bar.style.height = `${height}%`;
            bar.style.backgroundColor = i === parseInt(predictionElement.textContent) 
                ? '#4CAF50' 
                : '#2196F3';
            
            const digitLabel = document.createElement('div');
            digitLabel.className = 'confidence-bar-label';
            digitLabel.textContent = i;
            
            barContainer.appendChild(valueLabel);
            barContainer.appendChild(bar);
            barContainer.appendChild(digitLabel);
            confidenceBars.appendChild(barContainer);
        }
    }
    
    // Initialisation
    initCanvas();
    loadModel();
});
