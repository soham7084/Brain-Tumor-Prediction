from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from flask_wtf.file import FileField
from wtforms import SubmitField
from wtforms.validators import InputRequired
from prediction import predict_tumor_class, model, labels
import os


app = Flask(__name__)
app.config['SECRET_KEY'] = 'abcd1234'
app.config['IMAGE_FOLDER'] = 'static/image'

if not os.path.exists(app.config['IMAGE_FOLDER']):
    os.makedirs(app.config['IMAGE_FOLDER'])

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Predict")

@app.route('/', methods=['GET', 'POST'])
def upload():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        print(file.filename)
        filename = secure_filename(file.filename)
        input_file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
        file.save(input_file_path)
        predicted_class, confidence = predict_tumor_class(model, input_file_path, labels)
        return render_template('predict.html', file=file, filename=filename, predicted_class=predicted_class, confidence=confidence)
    
    return render_template('home.html', form=form)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['IMAGE_FOLDER'], filename)
    file.save(filepath)
    
    # Run the AI prediction
    predicted_class, confidence = predict_tumor_class(model, filepath, labels)
    
    # Return the results as JSON data for n8n
    return jsonify({
        "success": True,
        "filename": filename,
        "diagnosis": str(predicted_class),
        "confidence": float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)