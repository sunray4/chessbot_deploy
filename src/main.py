from .utils import chess_manager, GameContext
import torch
from huggingface_hub import hf_hub_download
import importlib.util

# Load the HuggingFace model once when the module loads
print("Loading chess model from HuggingFace...")

# Download model files from HuggingFace
model_path = hf_hub_download(repo_id="AubreeL/chess-bot", filename="chess_model.pth")
model_py_path = hf_hub_download(repo_id="AubreeL/chess-bot", filename="model.py")

# Import the model architecture from the downloaded file
spec = importlib.util.spec_from_file_location("chess_model", model_py_path)
if spec is None or spec.loader is None:
    raise ImportError("Could not load model.py from HuggingFace")
chess_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chess_model_module)

TinyPCN = chess_model_module.TinyPCN
encode_board = chess_model_module.encode_board
encode_move = chess_model_module.encode_move
decode_move = chess_model_module.decode_move

# Initialize and load the model
model = TinyPCN(board_channels=18, policy_size=4672)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

print("Model loaded successfully!")

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position
    print("Evaluating position...")
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Encode the board and run through the model
    board_tensor = encode_board(ctx.board).unsqueeze(0)
    
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
    
    print(f"Position evaluation: {value.item():.4f}")
    
    # Convert policy logits to probabilities
    policy_probs = torch.softmax(policy_logits[0], dim=0)
    
    # Map legal moves to their policy probabilities using proper move encoding
    move_probs = {}
    for move in legal_moves:
        try:
            # Use the model's move encoding to get the correct policy index
            move_idx = encode_move(move, ctx.board)
            if 0 <= move_idx < len(policy_probs):
                move_probs[move] = policy_probs[move_idx].item()
            else:
                # Fallback: assign very small probability to out-of-range moves
                move_probs[move] = 1e-10
        except Exception as e:
            # If encoding fails, assign minimal probability
            print(f"Warning: Failed to encode move {move.uci()}: {e}")
            move_probs[move] = 1e-10
    
    # Normalize probabilities to sum to 1.0
    total_prob = sum(move_probs.values())
    if total_prob > 0:
        move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
    else:
        # Fallback: uniform distribution if all probabilities are zero
        uniform_prob = 1.0 / len(legal_moves)
        move_probs = {move: uniform_prob for move in legal_moves}
    
    # Log probabilities for analysis
    ctx.logProbabilities(move_probs)
    
    # Sort moves by probability 
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    best_move = sorted_moves[0][0]
    
    print(f"Top 3 moves: {[(m.uci(), f'{p:.4f}') for m, p in sorted_moves[:3]]}")
    
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    # Clear any logged probabilities for the previous game
    try:
        ctx.logProbabilities({})
    except Exception:
        print("reset_func: failed to clear move probabilities")

    try:
        if hasattr(torch, "cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                print("reset_func: torch.cuda.empty_cache() failed or not available")
    except Exception:
        pass

    # Keep the model in eval mode in case anything toggled it
    try:
        if "model" in globals() and hasattr(model, "eval"):
            model.eval()
    except Exception:
        print("reset_func: failed to set model to eval()")

    print("reset_func: model/game reset completed")
