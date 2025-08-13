
from flask import Flask, request, render_template, send_from_directory, url_for, redirect 
import os
import subprocess
from White_Best_Move import process_and_move as process_white_move
from Black_Best_Move import process_and_move as process_black_move
from white_hint import process_chessboard
from white_hint import get_hint_from_best_move
import chess

# Initialize the Flask app and specify the template folder
app = Flask(__name__, template_folder=r"D:\Projects\Machine_learning\website\template")

# Paths to save uploaded images and results
UPLOAD_FOLDER = r"D:\Projects\Machine_learning\website\uploads"
RESULTS_FOLDER = r"D:\Projects\Machine_learning\website\results"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Ensure that the results and upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/select')
def select():
    return render_template('select.html')

@app.route('/hint')
def hint():
    return render_template('hint.html')

@app.route('/front')
def home():
    return render_template('front.html')


@app.route('/run_white', methods=['POST'])
def run_white():
    if 'file' not in request.files:
        return render_template('front.html', error="No file selected.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('front.html', error="No file selected.")
    
    if not allowed_file(file.filename):
        return render_template('front.html', error="Invalid file type. Please upload a PNG or JPG image.")
    
    try:
        # Generate a unique filename to prevent overwrites
        filename = f"upload_{os.path.splitext(file.filename)[0]}_{os.urandom(4).hex()}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image
        best_move, result_image_filename = process_white_move(file_path, app.config['RESULTS_FOLDER'])
        
        if best_move and result_image_filename:
            # Generate the URL for the result image
            result_image_url = url_for('send_result', filename=result_image_filename)
            return render_template('front.html',
                                result_image=result_image_url,
                                best_move=str(best_move),
                                success=True)
        else:
            return render_template('front.html',
                                error="Could not process the image.")
    except Exception as e:
        return render_template('front.html',
                            error=f"An error occurred: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/run_black', methods=['POST'])
def run_black():
    if 'file' not in request.files:
        return render_template('front.html', error="No file selected.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('front.html', error="No file selected.")
    
    if not allowed_file(file.filename):
        return render_template('front.html', error="Invalid file type. Please upload a PNG or JPG image.")
    
    try:
        # Generate a unique filename to prevent overwrites
        filename = f"upload_{os.path.splitext(file.filename)[0]}_{os.urandom(4).hex()}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image
        best_move, result_image_filename = process_black_move(file_path, app.config['RESULTS_FOLDER'])
        
        if best_move and result_image_filename:
            # Generate the URL for the result image
            result_image_url = url_for('send_result', filename=result_image_filename)
            return render_template('front.html',
                                result_image=result_image_url,
                                best_move=str(best_move),
                                success=True)
        else:
            return render_template('front.html',
                                error="Could not process the image.")
    except Exception as e:
        return render_template('front.html',
                            error=f"An error occurred: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route('/run_white_hint', methods=['POST'])
def run_white_hint():
    if 'file' not in request.files:
        return render_template('hint.html', error="No file selected.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('hint.html', error="No file selected.")
    
    if not allowed_file(file.filename):
        return render_template('hint.html', error="Invalid file type. Please upload a PNG or JPG image.")
    
    try:
        filename = f"upload_{os.path.splitext(file.filename)[0]}_{os.urandom(4).hex()}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        fen = process_chessboard(file_path)
        
        if fen:
            print(f"Processed FEN: {fen}")
            board = chess.Board(fen)
            
            hint = get_hint_from_best_move(board, depth=4)
            print(f"Hint: {hint}")
            
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Pass both hint and uploaded_image_url to the template7
            return render_template('hint.html', 
                                hint=hint,
                                uploaded_image_url=url_for('send_result', filename=os.path.basename(file_path)),
                                success=True)
        else:
            return render_template('hint.html', error="Could not analyze the chessboard position.")
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return render_template('hint.html', error=f"An error occurred: {str(e)}")

@app.route('/run_black_hint', methods=['POST'])
def run_black_hint():
    if 'file' not in request.files:
        return render_template('hint.html', error="No file selected.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('hint.html', error="No file selected.")
    
    if not allowed_file(file.filename):
        return render_template('hint.html', error="Invalid file type. Please upload a PNG or JPG image.")
    
    try:
        filename = f"upload_{os.path.splitext(file.filename)[0]}_{os.urandom(4).hex()}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        fen = process_chessboard(file_path)
        
        if fen:
            fen_parts = fen.split(' ')
            fen_parts[1] = 'b'
            fen = ' '.join(fen_parts)
            
            board = chess.Board(fen)
            hint = get_hint_from_best_move(board, depth=4)
            
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Pass both hint and uploaded_image_url to the template
            return render_template('hint.html', 
                                hint=hint,
                                uploaded_image_url=url_for('send_result', filename=os.path.basename(file_path)),
                                success=True)
        else:
            return render_template('hint.html', error="Could not analyze the chessboard position.")
            
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return render_template('hint.html', error=f"An error occurred: {str(e)}")

@app.route('/results/<filename>')
def send_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

