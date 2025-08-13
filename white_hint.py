import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import chess
import random
from pathlib import Path
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
    model_path = Path('C:/path_to_save_model/color_model.h5')
    return load_model(str(model_path))

def interpret_prediction(prediction):
    return 'White' if prediction[0][0] > 0.5 else 'Black'

def preprocess_input_image(imagefile):
    img = cv.imread(str(imagefile))
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
    model_path = Path('D:/Projects/Machine_learning/saved_model/chess_piece_classification_model.h5')
    model = load_model(str(model_path))
    resized_img_block = cv.resize(img_block, (85, 85))
    preprocessed_img = resized_img_block.astype("float32") / 255.0
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    prediction = model.predict(preprocessed_img)
    return interpret_piece_prediction(prediction)

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
    fen += " w KQkq - 0 1"
    
    return fen

def process_chessboard(imagefile):
    try:
        img_blocks = preprocess_input_image(imagefile)
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

    if white_pieces:
        print("White pieces:", ", ".join(white_pieces))
    if black_pieces:
        print("Black pieces:", ", ".join(black_pieces))

    fen = generate_fen(detected_pieces)
    print("\nFEN:", fen)

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
        max_eval = float('-inf')
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
base_hint_templates = { 
    "Pawn": [
        "2.3 Consider advancing this pawn to control the center.",
        "2.5 Use this pawn to support other pieces.",
        "1.8 Push the pawn to create a stronghold for your attack.",
        "3.1 Place this pawn in front of your king for protection.",
        "2.2 Advance the pawn to block the opponent's pieces.",
        "2.7 Use this pawn to gain space on the queenside.",
        "2.1 Push the pawn to gain tempo in your development.",
        "1.9 Consider a pawn push to open up the game.",
        "2.4 Place the pawn where it can restrict the opponent's knight.",
        "2.6 Consider advancing the pawn to challenge the opponent's center.",
        "2.8 Place the pawn where it can control a key square in the center.",
        "3.0 Push the pawn to create a passed pawn in the endgame.",
        "2.1 Move the pawn to limit the movement of your opponent's pieces.",
        "2.3 Consider pushing the pawn to initiate an attack on the flank.",
        "2.6 Use this pawn to defend against an incoming attack.",
    ],
    "Rook": [
        "3.1 Move the rook to an open file to maximize control.",
        "2.8 Use the rook to attack opponent's weak pawns.",
        "2.5 Consider doubling your rooks on an open file.",
        "3.0 Position your rook on the seventh rank to create pressure.",
        "2.7 Activate the rook to attack the opponent's back rank.",
        "2.4 Position your rook to support your central pawns.",
        "2.9 Keep your rook on an open file to limit the opponent's options.",
        "2.6 Place the rook behind your passed pawn to support its advancement.",
        "2.8 Use the rook to cut off the opponent's king from the center.",
        "2.3 Place the rook to defend your own king while attacking.",
        "3.1 Keep your rook active to control open lines and ranks.",
        "2.5 Use the rook to protect your pieces while launching a counterattack.",
        "2.9 Consider placing the rook in a strong defensive position to support your king.",
        "2.7 Keep the rook centralized to control more of the board.",
        "2.4 Use the rook in coordination with other pieces for a checkmate threat.",
    ],
    "Knight": [
        "3.2 Move the knight to fork two opponent pieces.",
        "2.6 Place the knight on a central square for more control.",
        "2.3 Consider moving the knight to protect other pieces.",
        "2.8 Position your knight to control the center of the board.",
        "2.9 Place the knight on a square where it can attack key squares.",
        "2.4 Keep the knight close to the king for extra defense.",
        "3.1 Move the knight to attack undefended pieces.",
        "2.7 Position the knight to control key diagonal squares.",
        "2.5 Use the knight to protect pawns in the center.",
        "2.2 Consider a knight jump to create a tactical threat.",
        "2.9 Place the knight where it can threaten your opponent's back rank.",
        "2.6 Move the knight to a position where it can attack your opponent's pawns.",
        "3.0 Use the knight to create a double attack on your opponent's pieces.",
        "2.8 Consider a knight maneuver to create a check or fork.",
        "2.4 Place the knight in an advanced position to gain more space.",
    ],
    "Bishop": [
        "3.1 Consider moving the bishop to control long diagonals.",
        "2.7 Place the bishop on an open diagonal for flexibility.",
        "2.5 Move the bishop to pin an opponent's knight.",
        "2.9 Use the bishop to control the center from a distance.",
        "2.4 Place the bishop on a light square to support your pawn structure.",
        "2.8 Place the bishop where it can attack the opponent's pawns.",
        "2.6 Use the bishop to defend against opponent's piece attacks.",
        "2.3 Position your bishop to support a kingside attack.",
        "3.0 Move the bishop to an open diagonal to control more space.",
        "2.7 Consider the bishop as a long-term defender in the endgame.",
        "2.9 Use the bishop to restrict your opponent's king's mobility.",
        "2.5 Move the bishop to a position where it can pin your opponent's pieces.",
        "2.8 Position the bishop to control multiple squares on the board.",
        "2.4 Consider using the bishop to control both sides of the board.",
        "2.6 Place the bishop where it can protect your pawns from attacks.",
    ],
    "Queen": [
        "3.2 Move the queen to support your other pieces.",
        "2.1 Avoid bringing the queen out too early in the game.",
        "2.7 Consider placing the queen on an open diagonal.",
        "2.9 Position the queen to control both the center and the flank.",
        "3.0 Use the queen to threaten the opponent's back rank.",
        "2.5 Coordinate your queen with the rooks to increase pressure.",
        "2.8 Move the queen to help create a checkmate threat.",
        "2.3 Position the queen to support a pawn promotion.",
        "2.6 Use the queen to restrict the opponent's king's mobility.",
        "3.1 Keep the queen active in the center to control key squares.",
        "2.4 Position the queen to support a kingside attack.",
        "2.7 Move the queen to a square where it can defend your pawns.",
        "2.9 Consider placing the queen on a long-range diagonal to control space.",
        "2.5 Use the queen to create a tactical threat with your knights or rooks.",
        "3.0 Keep the queen near the center to maximize its influence.",
    ],
    "King": [
        "2.8 Castle early to secure your king.",
        "3.1 Move the king to safety during the endgame.",
        "2.5 Keep the king shielded by pawns.",
        "2.9 In the endgame, activate the king to support pawn advancement.",
        "2.7 Use the king to support your pieces in the late game.",
        "2.6 Move the king toward the center in the endgame for more mobility.",
        "2.3 Position the king away from the edge to avoid attacks.",
        "2.8 Consider a king maneuver to help control the center.",
        "3.0 Ensure the king is well defended as the opponent launches an attack.",
        "2.4 Place the king in a safe, defended position during the middle game.",
        "2.7 Move the king to a corner where it can be protected by pawns.",
        "2.9 Keep the king close to your advanced pawns to support their promotion.",
        "2.5 Avoid placing the king on open ranks to reduce attack vulnerability.",
        "2.8 Consider positioning the king in the center during the late game for more mobility.",
        "2.6 Place the king where it can easily escape if the opponent initiates a checkmate threat.",
    ]
}


def get_hint_from_best_move(board, depth):
    best_move = None
    best_value = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    for move in board.legal_moves:
        board.push(move)
        board_value = minimax(board, depth - 1, alpha, beta, False)
        board.pop()

        if board_value > best_value:
            best_value = board_value
            best_move = move

    if best_move:
        piece_type = board.piece_type_at(best_move.from_square)
        if piece_type is None:
            return "No valid move found."
            
        piece_name = chess.piece_name(piece_type).capitalize()
        if piece_name in base_hint_templates:
            if best_value < -3 or best_value > 3:
                hint = random.choice(base_hint_templates[piece_name])
            else:
                hint_index = int((best_value + 4) // 0.5)
                hint_index = max(0, min(hint_index, len(base_hint_templates[piece_name]) - 1))
                hint = base_hint_templates[piece_name][hint_index]
        else:
            hint = "Consider developing your pieces and controlling the center."

        return f"{hint}"
    return "No valid move found."

def main():
    upload_folder = Path("D:/Projects/Machine_learning/website/uploads")
    
    if not upload_folder.exists():
        print("Upload folder not found.")
        return
    
    try:
        image_files = list(upload_folder.glob("*.png")) + list(upload_folder.glob("*.jpg"))
        
        if not image_files:
            print("No image files found.")
            return
        
        latest_image = max(image_files, key=lambda x: x.stat().st_mtime)
        fen = process_chessboard(latest_image)
        
        if fen:
            board = chess.Board(fen)
            hint = get_hint_from_best_move(board, depth=4)
            print("\nHint:", hint)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()