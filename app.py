from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from model import detect_objects_on_image, detect_objects_on_video

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    is_image = False
    class_counts = {}

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            ext = filename.split('.')[-1].lower()
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

            if ext in ['jpg', 'jpeg', 'png']:
                class_counts = detect_objects_on_image(file_path, result_path)
                is_image = True
            elif ext in ['mp4', 'avi', 'mov']:
                class_counts = detect_objects_on_video(file_path, result_path)
            else:
                return "Unsupported file format", 400

            filename = result_filename

    return render_template('index.html', filename=filename, is_image=is_image, class_counts=class_counts)

if __name__ == '__main__':
    app.run(debug=True)
