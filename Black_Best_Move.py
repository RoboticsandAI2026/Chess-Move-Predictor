import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import chess
from tensorflow.keras.models import load_model

def detect_piece_in_square(img_block):
    gray = cv.cvtColor(img_block, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    piece_color = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 500:
            square_color = determine_square_color(img_block)
            piece_color = classify_piece_color(img_block, square_color)
            return True, img_block, piece_color

    return False, None, None

def determine_square_color(img_block):
    average_color = cv.mean(img_block)[:3]
    average_intensity = sum(average_color) / len(average_color)
    return 'Black' if average_intensity < 127 else 'White'

def classify_piece_color(img_block, square_color):
    hsv = cv.cvtColor(img_block, cv.COLOR_RGB2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    black_mask = cv.inRange(hsv, lower_black, upper_black)
    white_mask = cv.inRange(hsv, lower_white, upper_white)
    black_count = cv.countNonZero(black_mask)
    white_count = cv.countNonZero(white_mask)

    if white_count >= 2 * black_count:
        return 'White'
    elif black_count >= 2 * white_count:
        return 'Black'
    return classify_using_model(img_block)

def classify_using_model(img_block):
    preprocessed_img = preprocess_for_model(img_block)
    model = load_your_model()
    prediction = model.predict(preprocessed_img)
    return interpret_prediction(prediction)

def preprocess_for_model(img):
    resized_img = cv.resize(img, (224, 224))
    normalized_img = resized_img / 255.0
    return np.expand_dims(normalized_img, axis=0)

def load_your_model():
    return load_model(r'color_model.h5')

def interpret_prediction(prediction):
    return 'White' if prediction[0][0] > 0.5 else 'Black'

def preprocess_input_image(imagefile):
    img = cv.imread(imagefile)
    if img is None:
        raise ValueError(f"Unable to read the image file: {imagefile}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    height, width, _ = img.shape
    square_size = height // 8
    img_blocks = []

    for i in range(8):
        for j in range(8):
            square = img[i * square_size:(i + 1) * square_size, j * square_size:(j + 1) * square_size]
            img_blocks.append(square)

    return img_blocks

def classify_piece_name(img_block):
    model = load_your_piece_model()
    resized_img_block = cv.resize(img_block, (85, 85))
    preprocessed_img = resized_img_block.astype("float32") / 255.0
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    prediction = model.predict(preprocessed_img)
    return interpret_piece_prediction(prediction)

def load_your_piece_model():
    return load_model('chess_piece_classification_model.h5')

def interpret_piece_prediction(prediction):
    piece_index = np.argmax(prediction)
    piece_classes = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']
    return piece_classes[piece_index]

def generate_fen(detected_pieces):
    board = [['' for _ in range(8)] for _ in range(8)]
    piece_to_fen = {
        'pawn': 'P', 'knight': 'N', 'bishop': 'B', 'rook': 'R', 'queen': 'Q', 'king': 'K'
    }
    
    for position, _, color, piece_name in detected_pieces:
        col = ord(position[0]) - ord('a')
        row = 8 - int(position[1])
        fen_char = piece_to_fen[piece_name.lower()]
        board[row][col] = fen_char.upper() if color == "White" else fen_char.lower()
    
    fen_parts = []
    for row in board:
        empty = 0
        row_fen = ''
        for cell in row:
            if cell == '':
                empty += 1
            else:
                if empty > 0:
                    row_fen += str(empty)
                    empty = 0
                row_fen += cell
        if empty > 0:
            row_fen += str(empty)
        fen_parts.append(row_fen)
    
    fen = '/'.join(fen_parts)
    fen += " b KQkq - 0 1"  # Black to move
    
    return fen

def process_chessboard(imagefile):
    try:
        img_blocks = preprocess_input_image(imagefile=imagefile)
    except ValueError as e:
        print(f"Error: {e}")
        return

    detected_pieces = []
    white_pieces = []
    black_pieces = []
    columns = 'abcdefgh'
    rows = range(8, 0, -1)

    for i, img_block in enumerate(img_blocks):
        piece_present, piece_image, piece_color = detect_piece_in_square(img_block)
        if piece_present:
            resized_piece_image = cv.resize(piece_image, (85, 85))
            piece_name = classify_piece_name(resized_piece_image)
            column = columns[i % 8]
            row = rows[i // 8]
            square_position = f"{column}{row}"
            detected_pieces.append((square_position, resized_piece_image, piece_color, piece_name))
            piece_info = f"{piece_name}-{square_position}"
            if piece_color == "White":
                white_pieces.append(piece_info)
            else:
                black_pieces.append(piece_info)

    original_img = cv.imread(imagefile)
    if original_img is not None:
        original_img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(original_img)
        plt.axis('off')
        plt.title("Original Chessboard")
        #plt.show()  # Show original image

    print("White pieces:", ", ".join(white_pieces))
    print("Black pieces:", ", ".join(black_pieces))

    fen = generate_fen(detected_pieces)
    print("\nFEN:", fen)

    if detected_pieces:
        print(f"\nDetected {len(detected_pieces)} squares with pieces.\n")
        r = len(detected_pieces)
        c = 1
        fig = plt.figure(figsize=(5, r))
        for i, (pos, img, color, name) in enumerate(detected_pieces):
            ax = fig.add_subplot(r, c, i + 1)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{pos}: {color} {name}")
        plt.tight_layout()  # Adjust layout to prevent overlap
        #plt.show()  # Show detected pieces
    else:
        print("No pieces detected on the chessboard.")

    return fen

def evaluate_board(board):
    material_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    score = 0
    for piece_type in material_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * material_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * material_values[piece_type]
    return score

def minimax(board, depth, alpha, beta, is_maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if is_maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(board, depth):
    best_move = None
    best_value = float('inf')  # Minimize for black
    alpha = -float('inf')
    beta = float('inf')

    for move in board.legal_moves:
        board.push(move)
        board_value = minimax(board, depth - 1, alpha, beta, True)
        board.pop()

        if board_value < best_value:
            best_value = board_value
            best_move = move

    return best_move

def get_square_image(img, square):
    file, rank = chess.square_file(square), chess.square_rank(square)
    height, width, _ = img.shape
    square_size = height // 8
    y = (7 - rank) * square_size
    x = file * square_size
    return img[y:y+square_size, x:x+square_size]

def put_square_image(img, square, square_img):
    file, rank = chess.square_file(square), chess.square_rank(square)
    height, width, _ = img.shape
    square_size = height // 8
    y = (7 - rank) * square_size
    x = file * square_size
    img[y:y+square_size, x:x+square_size] = square_img

def extract_piece(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv.drawContours(mask, contours, -1, (255), thickness=cv.FILLED)
        result = cv.bitwise_and(img, img, mask=mask)
        return result
    return img

def get_empty_square(img, is_white):
    height, width, _ = img.shape
    square_size = height // 8
    color = (240, 217, 181) if is_white else (181, 136, 99)  # RGB colors for white and black squares
    return np.full((square_size, square_size, 3), color, dtype=np.uint8)

def determine_square_color(square):
    file, rank = chess.square_file(square), chess.square_rank(square)
    return (file + rank) % 2 == 0  # True for white, False for black

def apply_move_to_image(img, move):
    from_square = move.from_square
    to_square = move.to_square
    
    # Get the images of the 'from' and 'to' squares
    from_img = get_square_image(img, from_square)
    to_img = get_square_image(img, to_square)
    
    # Determine the colors of the 'from' and 'to' squares
    from_is_white = determine_square_color(from_square)
    to_is_white = determine_square_color(to_square)
    
    # Extract the piece from the 'from' square
    piece_img = extract_piece(from_img)
    
    # Replace the 'to' square with an empty square of the appropriate color
    empty_to_square = get_empty_square(img, to_is_white)
    put_square_image(img, to_square, empty_to_square)
    
    # Place the extracted piece on the 'to' square
    put_square_image(img, to_square, cv.addWeighted(empty_to_square, 0.5, piece_img, 0.5, 0))
    
    # Replace the 'from' square with an empty square of the appropriate color
    empty_from_square = get_empty_square(img, from_is_white)
    put_square_image(img, from_square, empty_from_square)
    
    return img

# Process the chessboard and move the best piece
results_folder = r"D:\Projects\Machine_learning\website\results"  # Define your results folder path

def process_and_move(imagefile, results_folder):
    # Process the chessboard image and get FEN
    fen = process_chessboard(imagefile)

    if fen:
        # Initialize the chess board
        board = chess.Board(fen)
        depth = 4 # Depth of minimax search
        best_move = find_best_move(board, depth)
        print("Best move:", best_move)
        
        if best_move:
            # Load the original image
            original_img = cv.imread(imagefile)
            if original_img is not None:
                # Convert original image to RGB
                original_img_rgb = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
                
                # Create a copy and apply the best move
                updated_img = original_img.copy()
                updated_img = apply_move_to_image(updated_img, best_move)
                updated_img_rgb = cv.cvtColor(updated_img, cv.COLOR_BGR2RGB)

                # Create a figure with two subplots side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                
                # Plot original image
                ax1.imshow(original_img_rgb)
                ax1.axis('off')
                ax1.set_title("Original Chessboard")
                
                # Plot updated image
                ax2.imshow(updated_img_rgb)
                ax2.set_title(f"After Best Move: {best_move}")
                ax2.axis('off')
                
                # Add a main title
                plt.suptitle("Chess Analysis Result", fontsize=16)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save the comparison figure
                result_image_filename = f"{os.path.splitext(os.path.basename(imagefile))[0]}_comparison.png"
                save_path = os.path.join(results_folder, result_image_filename)
                
                # Save the plot
                save_plot(fig, save_path)
                print(f"Saved comparison image to: {save_path}")
                
                # Display the plot
                plt.show()
                
                return best_move, result_image_filename
            else:
                print("Failed to load the image file for updating.")
        else:
            print("No valid move found.")
    
    return None, None

def save_plot(fig, filename):
    try:
        fig.savefig(filename, bbox_inches='tight', dpi=300)  # Increased DPI for better quality
        print(f"Successfully saved plot to: {filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig)  # Close the figure after saving to free memory

# Example usage
if __name__ == "__main__":
    imagefile = r'path_to_your_chessboard_image.png'
    results_folder = r"D:\Projects\Machine_learning\website\results"  # Define your results folder path

    
    # Create results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    
    # Process the image and get the best move
    best_move, result_file = process_and_move(imagefile, results_folder)
    
    if best_move:
        print(f"Best move: {best_move}")
        print(f"Result saved as: {result_file}")
    else:
        print("No move was found or an error occurred.")