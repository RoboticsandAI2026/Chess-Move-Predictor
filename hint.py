import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

# Configure pytesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(img):
    """
    Extract text from an image using OCR.
    """
    # Convert to PIL Image
    pil_img = Image.fromarray(img)
    
    # Extract text using pytesseract
    text = pytesseract.image_to_string(pil_img, config='--psm 7')
    
    # Clean and return the text
    return text.strip()

def preprocess_for_ocr(img):
    """
    Preprocess image for better OCR results.
    """
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    # Apply thresholding
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Apply dilation to make text clearer
    kernel = np.ones((2,2), np.uint8)
    dilated = cv.dilate(thresh, kernel, iterations=1)
    
    return dilated

def detect_piece_in_square(img_block):
    """
    Modified version of detect_piece_in_square that includes text detection.
    """
    # Original piece detection code
    gray = cv.cvtColor(img_block, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    piece_color = None
    detected_text = None
    
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 500:
            # Determine colors
            square_color = determine_square_color(img_block)
            piece_color = classify_piece_color(img_block, square_color)
            
            # Preprocess for OCR
            processed_img = preprocess_for_ocr(img_block)
            
            # Extract text
            detected_text = extract_text_from_image(processed_img)
            
            return True, img_block, piece_color, detected_text

    return False, None, None, None

def process_chessboard(imagefile):
    """
    Modified process_chessboard function that includes text recognition.
    """
    img_blocks = preprocess_input_image(imagefile=imagefile)
    detected_pieces = []
    white_pieces = []
    black_pieces = []

    columns = 'abcdefgh'
    rows = range(8, 0, -1)

    for i, img_block in enumerate(img_blocks):
        piece_present, piece_image, piece_color, detected_text = detect_piece_in_square(img_block)
        if piece_present:
            resized_piece_image = cv.resize(piece_image, (85, 85))
            piece_name = classify_piece_name(resized_piece_image)
            
            column = columns[i % 8]
            row = rows[i // 8]
            square_position = f"{column}{row}"

            # Include detected text in the output
            piece_info = {
                'position': square_position,
                'image': resized_piece_image,
                'color': piece_color,
                'piece_name': piece_name,
                'detected_text': detected_text if detected_text else "No text detected"
            }
            
            detected_pieces.append(piece_info)
            
            # Format piece information including text
            piece_text = f"{piece_name}-{square_position}"
            if detected_text:
                piece_text += f" ({detected_text})"
                
            if piece_color == "White":
                white_pieces.append(piece_text)
            else:
                black_pieces.append(piece_text)

    # Display original image
    original_img = cv.imread(imagefile)
    original_img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(original_img)
    plt.axis('off')
    plt.title("Original Chessboard")
    plt.show()

    # Print formatted output with text
    print("\nWhite pieces:", ", ".join(white_pieces))
    print("Black pieces:", ", ".join(black_pieces))

    # Generate and print FEN
    fen = generate_fen([(p['position'], p['image'], p['color'], p['piece_name']) for p in detected_pieces])
    print("\nFEN:", fen)

    # Visualize detected pieces with text
    if detected_pieces:
        print(f"\nDetected {len(detected_pieces)} squares with pieces.\n")
        r = len(detected_pieces)
        c = 1
        fig = plt.figure(figsize=(5, r))
        for i, piece in enumerate(detected_pieces):
            ax = fig.add_subplot(r, c, i + 1)
            ax.imshow(piece['image'])
            ax.axis('off')
            title = f"{piece['position']}: {piece['color']} {piece['piece_name']}"
            if piece['detected_text']:
                title += f"\nText: {piece['detected_text']}"
            ax.set_title(title)
        plt.show()
    else:
        print("No pieces detected on the chessboard.")