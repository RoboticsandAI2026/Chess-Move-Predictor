import cv2 as cv
import numpy as np
import chess
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import io
import streamlit as st
import random
import os

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide", 
    page_title="Chess Position Analyzer",
    page_icon="♟️",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Mode selection cards */
    .mode-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid transparent;
        background-clip: padding-box;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        text-align: center;
        margin: 1rem 0;
    }
    
    .mode-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        border: 3px solid #667eea;
    }
    
    .mode-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .mode-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 1rem;
    }
    
    .mode-description {
        color: #666;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    
    /* Perspective selection */
    .perspective-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
    }
    
    .perspective-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(245, 87, 108, 0.4);
    }
    
    .perspective-card.white {
        background: linear-gradient(135deg, #e0e0e0 0%, #f5f5f5 100%);
        color: #333;
    }
    
    .perspective-card.black {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
    }
    
    /* Progress indicators */
    .progress-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }
    
    .progress-step {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #e0e0e0;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 1rem;
        font-weight: bold;
        color: #666;
    }
    
    .progress-step.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .progress-step.completed {
        background: #4CAF50;
        color: white;
    }
    
    .progress-line {
        width: 60px;
        height: 3px;
        background: #e0e0e0;
    }
    
    .progress-line.completed {
        background: #4CAF50;
    }
    
    /* Results styling */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .result-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .result-title .icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    /* Image containers */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Navigation buttons */
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .nav-button.secondary {
        background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
    }
    
    .status-info {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(255, 152, 0, 0.3);
    }
    
    /* Piece gallery */
    .piece-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .piece-item {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .piece-item:hover {
        transform: scale(1.05);
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.8s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'mode_selection'
if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = None
if 'fen' not in st.session_state:
    st.session_state.fen = None
if 'board' not in st.session_state:
    st.session_state.board = None
if 'best_move' not in st.session_state:
    st.session_state.best_move = None
if 'hint' not in st.session_state:
    st.session_state.hint = None
if 'perspective' not in st.session_state:
    st.session_state.perspective = None
if 'selected_image_path' not in st.session_state:
    st.session_state.selected_image_path = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# Enhanced Title Section
st.markdown("""
<div class="main-header fade-in">
    <h1>♟️ Chess Position Analyzer</h1>
    <p>Advanced AI-powered chess analysis and move recommendations</p>
</div>
""", unsafe_allow_html=True)

# Default chess position images (you can add your own image paths here)
DEFAULT_IMAGES = {
    "Opening Position": {
        "path": r"fourth.png",
        "icon": "🎯"
    },
    "Middle Game 1": {
        "path": r"two.png",
        "icon": "⚔️"
    },
    "Middle Game 2": {
        "path": r"one.png",
        "icon": "🎪"
    },
    "Endgame Position": {
        "path": r"first.png",
        "icon": "🏁"
    },
    "Tactical Puzzle": {
        "path": r"five.png",
        "icon": "🧩"
    }
}

# Base hint templates for each piece type (keeping original)
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

def display_progress_indicator(current_step):
    """Display a visual progress indicator"""
    steps = ["Mode", "Setup", "Results"]
    step_numbers = {
        'mode_selection': 0,
        'perspective_selection': 1,
        'analysis_results': 2
    }
    
    current_index = step_numbers.get(current_step, 0)
    
    progress_html = '<div class="progress-indicator">'
    
    for i, step in enumerate(steps):
        # Step circle
        if i < current_index:
            step_class = "progress-step completed"
        elif i == current_index:
            step_class = "progress-step active"
        else:
            step_class = "progress-step"
        
        progress_html += f'<div class="{step_class}">{i+1}</div>'
        
        # Progress line
        if i < len(steps) - 1:
            line_class = "progress-line completed" if i < current_index else "progress-line"
            progress_html += f'<div class="{line_class}"></div>'
    
    progress_html += '</div>'
    
    # Step labels
    progress_html += '<div style="display: flex; justify-content: center; margin-top: 0.5rem;">'
    for i, step in enumerate(steps):
        color = "#4CAF50" if i < current_index else "#667eea" if i == current_index else "#999"
        progress_html += f'<div style="width: 100px; text-align: center; color: {color}; font-weight: bold; margin: 0 1rem;">{step}</div>'
    
    progress_html += '</div>'
    
    st.markdown(progress_html, unsafe_allow_html=True)

def go_back_to_mode_selection():
    """Reset to mode selection page"""
    st.session_state.page = 'mode_selection'
    st.session_state.selected_mode = None
    st.session_state.perspective = None
    st.session_state.selected_image_path = None
    st.session_state.uploaded_image = None

def show_mode_selection_page():
    """Display the enhanced mode selection page"""
    display_progress_indicator('mode_selection')
    
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # Create two columns for mode selection
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="mode-card slide-in">
            <div class="mode-icon">🏆</div>
            <div class="mode-title">Best Move Analysis</div>
            <div class="mode-description">
                Get the optimal move recommendation using advanced minimax algorithm 
                with alpha-beta pruning. Perfect for finding the strongest tactical moves.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Analyze Best Move", key="best_move_btn", use_container_width=True, type="primary"):
            st.session_state.selected_mode = "best_move"
            st.session_state.page = 'perspective_selection'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="mode-card slide-in">
            <div class="mode-icon">💡</div>
            <div class="mode-title">Strategic Hints</div>
            <div class="mode-description">
                Receive intelligent hints and strategic guidance tailored to your position. 
                Improve your understanding with contextual advice.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Get Strategic Hint", key="hint_btn", use_container_width=True, type="secondary"):
            st.session_state.selected_mode = "hint"
            st.session_state.page = 'perspective_selection'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("---")
    st.markdown("### ✨ Key Features")
    
    feature_cols = st.columns(4)
    features = [
        ("🔍", "AI Vision", "Advanced computer vision for piece detection"),
        ("🧠", "Smart Analysis", "Deep learning algorithms for position evaluation"),
        ("⚡", "Real-time", "Instant analysis and move suggestions"),
        ("🎨", "Intuitive UI", "Beautiful and user-friendly interface")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with feature_cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-weight: bold; color: #333; margin-bottom: 0.5rem;">{title}</div>
                <div style="color: #666; font-size: 0.9rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

def show_perspective_selection_page():
    """Display enhanced perspective selection and configuration page"""
    display_progress_indicator('perspective_selection')
    
    # Header with back button
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Back", key="back_btn", use_container_width=True):
            go_back_to_mode_selection()
            st.rerun()
    
    with col2:
        mode_title = st.session_state.selected_mode.replace('_', ' ').title()
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #667eea; margin: 0;">🎨 {mode_title} Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Perspective Selection with enhanced cards
    st.markdown("### 🎯 Choose Your Perspective")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="perspective-card white">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚪</div>
            <div style="font-size: 1.5rem; font-weight: bold;">White to Move</div>
            <div style="margin-top: 0.5rem;">Play as White pieces</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select White", key="white_btn", use_container_width=True):
            st.session_state.perspective = "White"
    
    with col2:
        st.markdown("""
        <div class="perspective-card black">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚫</div>
            <div style="font-size: 1.5rem; font-weight: bold;">Black to Move</div>
            <div style="margin-top: 0.5rem;">Play as Black pieces</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Black", key="black_btn", use_container_width=True):
            st.session_state.perspective = "Black"
    
    if st.session_state.perspective:
        st.markdown(f"""
        <div class="status-success fade-in">
            ✅ Selected: <strong>{st.session_state.perspective}</strong> to move
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Image Selection Section
        st.markdown("### 📸 Choose Your Chess Position")
        
        # Tabbed interface for image selection
        tab1, tab2 = st.tabs(["📁 Upload Custom", "🎯 Default Positions"])
        
        with tab1:
            st.markdown("""
            <div class="result-card">
                <div class="result-title">
                    <span class="icon">📤</span>
                    Upload Your Chessboard Image
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Drop your chessboard image here or click to browse",
                type=["png", "jpg", "jpeg"],
                help="Upload a clear, well-lit image of a chessboard position",
                key="file_uploader"
            )
            if uploaded_file is not None:
                st.session_state.uploaded_image = uploaded_file
                st.session_state.selected_image_path = None
                
                # Show preview
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(uploaded_file, caption="Uploaded Image Preview", use_column_width=True)
                    st.markdown('<div class="status-success">✅ Image uploaded successfully!</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="result-card">
                <div class="result-title">
                    <span class="icon">🎯</span>
                    Select from Default Positions
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced default image selection
            cols = st.columns(len(DEFAULT_IMAGES))
            
            for idx, (name, info) in enumerate(DEFAULT_IMAGES.items()):
                with cols[idx % len(cols)]:
                    icon = info["icon"]
                    path = info["path"]
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; margin: 0.5rem 0;">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                        <div style="font-weight: bold; margin-bottom: 0.5rem;">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Select {name}", key=f"default_{idx}", use_container_width=True):
                        st.session_state.selected_image_path = path
                        st.session_state.uploaded_image = None
                        st.markdown(f'<div class="status-success">✅ Selected: {name}</div>', unsafe_allow_html=True)
        
        # Analysis Settings in an expandable section
        with st.expander("⚙️ Advanced Settings", expanded=True):
            st.markdown("**Search Depth Configuration**")
            depth = st.slider(
                "Analysis Depth", 
                min_value=1, 
                max_value=5, 
                value=3,
                help="Higher depth provides more accurate analysis but takes longer to compute"
            )
            
            # Depth explanation
            depth_info = {
                1: "⚡ Quick analysis - Basic move evaluation",
                2: "🔍 Standard analysis - Good balance of speed and accuracy", 
                3: "🎯 Recommended - Thorough analysis with good performance",
                4: "🧠 Deep analysis - More accurate but slower",
                5: "🔬 Maximum depth - Highest accuracy, longest processing time"
            }
            
            st.info(depth_info[depth])
        
        # Enhanced Analyze button
        if (st.session_state.uploaded_image is not None or st.session_state.selected_image_path is not None):
            st.markdown("---")
            st.markdown("### 🚀 Ready to Analyze!")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Start Analysis", key="analyze_btn", use_container_width=True, type="primary"):
                    with st.spinner("🤖 Analyzing position..."):
                        st.session_state.page = 'analysis_results'
                        process_analysis(depth)
                    st.rerun()
        else:
            st.markdown("""
            <div class="status-info">
                ℹ️ Please select an image source to continue
            </div>
            """, unsafe_allow_html=True)

def load_image():
    """Load image from either uploaded file or default path"""
    if st.session_state.uploaded_image is not None:
        file_bytes = np.asarray(bytearray(st.session_state.uploaded_image.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        return img
    elif st.session_state.selected_image_path is not None:
        if os.path.exists(st.session_state.selected_image_path):
            img = cv.imread(st.session_state.selected_image_path)
            return img
        else:
            # If default image doesn't exist, create a placeholder
            st.warning(f"Default image not found: {st.session_state.selected_image_path}")
            # Create a placeholder chessboard image
            return create_placeholder_chessboard()
    return None

def create_placeholder_chessboard():
    """Create a placeholder chessboard image for demonstration"""
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    square_size = 100
    
    # Create alternating squares
    for i in range(8):
        for j in range(8):
            color = (240, 217, 181) if (i + j) % 2 == 0 else (181, 136, 99)
            y1, y2 = i * square_size, (i + 1) * square_size
            x1, x2 = j * square_size, (j + 1) * square_size
            img[y1:y2, x1:x2] = color
    
    # Add some text indicating it's a placeholder
    cv.putText(img, "Sample Chessboard", (200, 400), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv.putText(img, "(Add your images)", (220, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img

def process_analysis(depth):
    """Process the chess analysis"""
    try:
        img = load_image()
        if img is None:
            st.error("Could not load image")
            return
        
        fen, white_pieces, black_pieces, detected_pieces = process_chessboard(img)
        
        if fen:
            try:
                st.session_state.fen = fen
                st.session_state.board = chess.Board(fen)
                
                if st.session_state.selected_mode == "best_move":
                    best_move = find_best_move(st.session_state.board, depth, st.session_state.perspective)
                    st.session_state.best_move = best_move
                
                elif st.session_state.selected_mode == "hint":
                    hint = get_hint_from_best_move(st.session_state.board, depth, st.session_state.perspective)
                    st.session_state.hint = hint
                
                # Store analysis results
                st.session_state.white_pieces = white_pieces
                st.session_state.black_pieces = black_pieces
                st.session_state.detected_pieces = detected_pieces
                st.session_state.original_img = img
                
            except ValueError as e:
                st.error(f"Invalid FEN notation generated: {e}")
        else:
            st.error("Failed to generate FEN notation from the image")
            
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")

def show_analysis_results():
    """Display enhanced analysis results"""
    display_progress_indicator('analysis_results')
    
    # Enhanced header with navigation
    col1, col2, col3, col4 = st.columns([1, 1, 4, 1])
    
    with col1:
        if st.button("← Back", key="back_to_config", use_container_width=True):
            st.session_state.page = 'perspective_selection'
            st.rerun()
    
    with col2:
        if st.button("🏠 Start Over", key="start_over", use_container_width=True):
            go_back_to_mode_selection()
            st.rerun()
    
    with col3:
        mode_title = st.session_state.selected_mode.replace('_', ' ').title()
        st.markdown(f"""
        <div style="text-align: center;">
            <h2 style="color: #667eea; margin: 0;">📊 {mode_title} Results</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display results based on mode
    if hasattr(st.session_state, 'original_img'):
        display_enhanced_results(
            st.session_state.original_img,
            st.session_state.fen,
            st.session_state.white_pieces,
            st.session_state.black_pieces,
            st.session_state.detected_pieces,
            st.session_state.selected_mode,
            st.session_state.perspective,
            best_move=getattr(st.session_state, 'best_move', None),
            hint=getattr(st.session_state, 'hint', None)
        )
    else:
        st.markdown("""
        <div class="status-warning">
            ⚠️ No analysis results available. Please run analysis again.
        </div>
        """, unsafe_allow_html=True)

# Chess analysis functions (keeping all original functions)
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
            square_color = determine_square_color_from_image(img_block)
            piece_color = classify_piece_color(img_block, square_color)
            return True, img_block, piece_color

    return False, None, None

def determine_square_color_from_image(img_block):
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
    try:
        preprocessed_img = preprocess_for_model(img_block)
        model = load_your_model()
        if model:
            prediction = model.predict(preprocessed_img)
            return interpret_prediction(prediction)
    except:
        pass
    return 'White'  # Default fallback

def preprocess_for_model(img):
    resized_img = cv.resize(img, (224, 224))
    normalized_img = resized_img / 255.0
    return np.expand_dims(normalized_img, axis=0)

def load_your_model():
    try:
        return load_model(r'https://drive.google.com/file/d/1GhZpHqWBIzUG2psayZOK9mdsK0osuWSJ/view?usp=sharing')
    except:
        return None

def interpret_prediction(prediction):
    return 'White' if prediction[0][0] > 0.5 else 'Black'

def preprocess_input_image(img):
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
    try:
        model = load_your_piece_model()
        if model is None:
            return 'pawn'
        resized_img_block = cv.resize(img_block, (85, 85))
        preprocessed_img = resized_img_block.astype("float32") / 255.0
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        prediction = model.predict(preprocessed_img)
        return interpret_piece_prediction(prediction)
    except:
        return 'pawn'

def load_your_piece_model():
    try:
        return load_model(r'https://drive.google.com/file/d/11Dxh6WoB2xqHquRVLatP67bCy6ZCRV24/view?usp=sharing')
    except:
        return None

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

def find_best_move(board, depth, perspective):
    best_move = None
    if perspective == "White":
        best_value = -float('inf')
    else:
        best_value = float('inf')
    
    alpha = -float('inf')
    beta = float('inf')

    for move in board.legal_moves:
        board.push(move)
        board_value = minimax(board, depth - 1, alpha, beta, perspective == "Black")
        board.pop()

        if perspective == "White":
            if board_value > best_value:
                best_value = board_value
                best_move = move
        else:
            if board_value < best_value:
                best_value = board_value
                best_move = move
    
    return best_move

def get_hint_from_best_move(board, depth, perspective):
    best_move = None
    if perspective == "White":
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            
            if board_value > best_value:
                best_value = board_value
                best_move = move
    else:
        best_value = float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            board_value = -minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            
            if board_value < best_value:
                best_value = board_value
                best_move = move

    if best_move:
        piece_type = board.piece_type_at(best_move.from_square)
        piece_name = chess.piece_name(piece_type).capitalize()

        if piece_name in base_hint_templates:
            if best_value < -3 or best_value > 3:
                hint = random.choice(base_hint_templates[piece_name])
            else:
                hint_index = int((best_value + 4) // 0.5)
                hint_index = max(0, min(hint_index, len(base_hint_templates[piece_name]) - 1))
                hint = base_hint_templates[piece_name][hint_index]
        else:
            hint = "Think about developing or positioning your pieces."

        hint += f" (Evaluation: {best_value:.2f})"
        return hint
    return "No valid move found."

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
    color = (240, 217, 181) if is_white else (181, 136, 99)
    return np.full((square_size, square_size, 3), color, dtype=np.uint8)

def determine_chess_square_color(square):
    file, rank = chess.square_file(square), chess.square_rank(square)
    return (file + rank) % 2 == 0

def apply_move_to_image(img, move):
    from_square = move.from_square
    to_square = move.to_square
    
    from_img = get_square_image(img, from_square)
    to_img = get_square_image(img, to_square)
    
    from_is_white = determine_chess_square_color(from_square)
    to_is_white = determine_chess_square_color(to_square)
    
    piece_img = extract_piece(from_img)
    
    empty_to_square = get_empty_square(img, to_is_white)
    put_square_image(img, to_square, empty_to_square)
    
    put_square_image(img, to_square, cv.addWeighted(empty_to_square, 0.5, piece_img, 0.5, 0))
    
    empty_from_square = get_empty_square(img, from_is_white)
    put_square_image(img, from_square, empty_from_square)
    
    return img

def process_chessboard(img):
    try:
        img_blocks = preprocess_input_image(img)
    except ValueError as e:
        st.error(f"Error: {e}")
        return None, [], [], []

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

    fen = generate_fen(detected_pieces)
    return fen, white_pieces, black_pieces, detected_pieces

def display_enhanced_results(original_img, fen, white_pieces, black_pieces, detected_pieces, mode, perspective, best_move=None, hint=None):
    """Enhanced results display with better styling and organization"""
    
    # Main result section
    if mode == "best_move" and best_move:
        # Best Move Analysis Results
        st.markdown("""
        <div class="result-card fade-in">
            <div class="result-title">
                <span class="icon">🎯</span>
                Best Move Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(original_img, caption="🔍 Original Position", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            moved_img = apply_move_to_image(original_img.copy(), best_move)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(moved_img, caption=f"✅ After Best Move: {best_move}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Best move highlight
        st.markdown(f"""
        <div class="status-success fade-in">
            🏆 <strong>Recommended Move for {perspective}:</strong> {best_move}
        </div>
        """, unsafe_allow_html=True)
        
    elif mode == "hint":
        # Hint Analysis Results
        st.markdown("""
        <div class="result-card fade-in">
            <div class="result-title">
                <span class="icon">💡</span>
                Strategic Analysis & Hints
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(original_img, caption="🔍 Current Position", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-top: 0;">
                <div class="result-title" style="color: white; border: none;">
                    <span class="icon">🎯</span>
                    Strategic Guidance for {perspective}
                </div>
                <div style="font-size: 1.1rem; line-height: 1.6; padding: 1rem 0;">
                    {hint}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Piece Detection Summary
    st.markdown("---")
    st.markdown("""
    <div class="result-card slide-in">
        <div class="result-title">
            <span class="icon">📋</span>
            Piece Detection Summary
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="result-card" style="text-align: center; background: linear-gradient(135deg, #e0e0e0 0%, #f5f5f5 100%);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚪</div>
            <div style="font-weight: bold; color: #333; margin-bottom: 0.5rem;">White Pieces</div>
            <div style="color: #666;">{', '.join(white_pieces) if white_pieces else 'None detected'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="result-card" style="text-align: center; background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color: white;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚫</div>
            <div style="font-weight: bold; margin-bottom: 0.5rem;">Black Pieces</div>
            <div style="opacity: 0.9;">{', '.join(black_pieces) if black_pieces else 'None detected'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="result-card" style="text-align: center; background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
            <div style="font-weight: bold; margin-bottom: 0.5rem;">Total Detected</div>
            <div style="font-size: 1.5rem;">{len(detected_pieces)} pieces</div>
        </div>
        """, unsafe_allow_html=True)
    
    # FEN Notation Section
    st.markdown("""
    <div class="result-card slide-in">
        <div class="result-title">
            <span class="icon">🔤</span>
            Position Notation (FEN)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.code(fen, language="text")
    
    # Detailed Piece Analysis
    if detected_pieces:
        st.markdown("---")
        st.markdown("""
        <div class="result-card slide-in">
            <div class="result-title">
                <span class="icon">🔍</span>
                Detailed Piece Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="status-info">
            📊 Successfully detected and analyzed {len(detected_pieces)} pieces on the board
        </div>
        """, unsafe_allow_html=True)
        
        # Piece gallery with enhanced styling
        st.markdown('<div class="piece-gallery">', unsafe_allow_html=True)
        
        # Create responsive columns for pieces
        cols_per_row = 6
        for i in range(0, len(detected_pieces), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, (pos, img, color, name) in enumerate(detected_pieces[i:i+cols_per_row]):
                with cols[j]:
                    # Color-coded piece display
                    bg_color = "linear-gradient(135deg, #e0e0e0 0%, #f5f5f5 100%)" if color == "White" else "linear-gradient(135deg, #2c3e50 0%, #34495e 100%)"
                    text_color = "#333" if color == "White" else "white"
                    
                    st.markdown(f"""
                    <div class="piece-item" style="background: {bg_color}; color: {text_color};">
                        <div style="font-weight: bold; margin-bottom: 0.5rem; font-size: 0.9rem;">{pos}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(img, caption=f"{color} {name}", width=100)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons for next steps
    st.markdown("---")
    st.markdown("### 🚀 What's Next?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Analyze Different Position", key="new_analysis", use_container_width=True):
            st.session_state.page = 'perspective_selection'
            st.rerun()
    
    with col2:
        other_mode = "hint" if mode == "best_move" else "best_move"
        other_mode_title = other_mode.replace('_', ' ').title()
        if st.button(f"💡 Try {other_mode_title}", key="switch_mode", use_container_width=True):
            st.session_state.selected_mode = other_mode
            st.session_state.page = 'perspective_selection'
            st.rerun()
    
    with col3:
        if st.button("🏠 Main Menu", key="main_menu", use_container_width=True):
            go_back_to_mode_selection()
            st.rerun()

# Main page routing
def main():
    """Main application routing"""
    
    if st.session_state.page == 'mode_selection':
        show_mode_selection_page()
    
    elif st.session_state.page == 'perspective_selection':
        show_perspective_selection_page()
    
    elif st.session_state.page == 'analysis_results':
        show_analysis_results()
    
    else:
        # Fallback to mode selection
        st.session_state.page = 'mode_selection'
        show_mode_selection_page()

# Run the main application
if __name__ == "__main__":

    main()

