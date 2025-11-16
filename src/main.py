from .utils import chess_manager, GameContext
import torch
from huggingface_hub import hf_hub_download

# Load the HuggingFace model once when the module loads
print("Loading chess model from HuggingFace...")

# Download model files from HuggingFace
model_path = hf_hub_download(repo_id="AubreeL/chess-bot", filename="chess_model.pth")
model_py_path = hf_hub_download(repo_id="AubreeL/chess-bot", filename="model.py")

# Import the model architecture from the downloaded file
import importlib.util
spec = importlib.util.spec_from_file_location("chess_model", model_py_path)
if spec is None or spec.loader is None:
    raise ImportError("Could not load model.py from HuggingFace")
chess_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chess_model_module)

TinyPCN = chess_model_module.TinyPCN
encode_board = chess_model_module.encode_board

# Initialize and load the model
model = TinyPCN(board_channels=18, policy_size=4672)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

print("Model loaded successfully!")

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Evaluating position with neural network...")
    
    # Get legal moves
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Encode the board and run through the model
    board_tensor = encode_board(ctx.board).unsqueeze(0)
    
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
    
    print(f"Position evaluation: {value.item():.4f}")
    
    # Convert policy logits to probabilities for legal moves
    # Note: The model outputs 4672 possible move encodings
    # We need to map legal moves to their policy indices and get probabilities
    
    # For now, use softmax on policy logits and select based on legal moves
    policy_probs = torch.softmax(policy_logits[0], dim=0)
    
    # Map legal moves to probabilities
    # This is a simplified approach - you may need to implement proper move encoding
    move_probs = {}
    for move in legal_moves:
        # Get a heuristic probability (this is simplified)
        # In a full implementation, you'd map each move to its policy index
        move_uci = move.uci()
        # Simple heuristic: use position in legal moves list
        idx = hash(move_uci) % len(policy_probs)
        move_probs[move] = policy_probs[idx].item()
    
    # Normalize probabilities
    total_prob = sum(move_probs.values())
    if total_prob > 0:
        move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
    
    # Log probabilities for analysis
    ctx.logProbabilities(move_probs)
    
    # Select move with highest probability
    best_move = max(move_probs.items(), key=lambda x: x[1])[0]
    
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
