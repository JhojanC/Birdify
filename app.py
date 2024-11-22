from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from io import BytesIO
from PIL import Image  # Use PIL for image processing in memory

# Class names list
names = ['Anser anser - Ganso Comun', 'Cerceta Alas Azulesf', 'Cerceta Andina',
          'Cerceta Canela', 'Cerceta sp', 'Chachalaca Colombiana', 'Chauna torquata - Chaja Norteno',
        'Colin Crestudo', 'Columbidae sp', 'Corcovado Castano', 'Crypturellus erythropus - Tinamu Patirrojo',
        'Dendrocygna autumnalis - Pijije Alas Blancas', 'Dendrocygna bicolor - Pijije Canelo',
         'Dendrocygna viduata - Pijije Cariblanco', 'Flamenco Americano', 'Gallo Bankiva',
          'Ganso Careto Mayor - Ganso Careto Mayor', 'Ganso del Orinoco', 'Ganso Egipcio',
            'Ganso sp. (Domestico)', 'Guajolote Norteno', 'Nothocercus bonapartei - Tinamu Serrano',
              'Nothocercus julius - Tinamu Cabecirrojo', 'Nothura minor - Tinamu Menor', 'Paloma Arroyera',
                'Paloma Cabeza Gris', 'Paloma Canela', 'Paloma Colorada', 'Paloma Comun', 'Paloma Encinera',
                  'Paloma Escamosa', 'Paloma Montaraz de Tolima', 'Paloma Perdiz Gorjiblanca', 'Paloma Plomiza',
                    'Pato Andino', 'Pato Boludo Menor', 'Pato Chalcuan', 'Pato Crestudo Americano', 'Pato Cucharon Norteno',
                      'Pato de Collar', 'Pato Enmascarado', 'Pato Real', 'Pato Real x de Collar (hibrido)',
                        'Pato Torrentero', 'Pava Aburria', 'Pava Andina', 'Pava del Cauca', 'Pava Falcialar',
                          'Pavo Real Comun', 'Pintada Comun', 'Tinamus major - Tinamu Mayor', 'Tortola Azul', 
                          'Tortolita Canela', 'Tortolita Pecho Liso', 'Tortolita Pico Rojo', 'Zambullidor Menor',
                            'Zambullidor Pico Grueso', 'Zenaida Torcaza']

# Initialize the app
app = Flask(__name__)
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# Allow CORS for all origins
CORS(app)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'best_model.keras')
modelt = load_model(model_path)

# Check if file type is allowed
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Endpoint to classify image
@app.route('/classify', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            # Open the image directly from memory
            image = Image.open(BytesIO(file.read()))
            image = image.resize((224, 224))  # Resize to model input size
            image_array = np.expand_dims(preprocess_input(np.array(image)), axis=0)

            # Get predictions
            preds = modelt.predict(image_array)
            predicted_class_index = np.argmax(preds)
            if not (predicted_class_index < len(names)):
                return jsonify({"error": "Predicted index is out of range."}), 500
            predicted_class_name = names[predicted_class_index]
            confidence_percentage = preds[0][predicted_class_index] * 100

            return jsonify({
                "message": f'Clase Identificada: {predicted_class_name}, Coincidencia: {confidence_percentage:.2f}%',
            }), 200

        except Exception as e:
            return jsonify({"error": f"Error processing the image: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type"}), 400

# Custom 404 error handler
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

# Serve interface
@app.route('/')
def serve_interface():
    return send_from_directory('.', 'index.html')

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080)) 
    app.run(host='0.0.0.0', port=port)
