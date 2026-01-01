import torch
import json
import torch.nn.functional as F
from model import SkinTransformer
from tqdm import tqdm

# CONFIG
MODEL_PATH = "checkpoints2/skin_model_v2_ep50.pth"
OUTPUT_JSON = "smart_skins.json"
NUM_GEN = 1
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- REVERSE MAPPING HELPERS ---
def token_to_val(token):
    if 100 <= token < 300: return ("id", token - 100)
    if 300 <= token < 400: return ("x", token - 300 - 32)
    if 400 <= token < 500: return ("y", token - 400 - 32)
    if 500 <= token < 600: return ("color", int(((token - 500) / 31.0) * 255))
    if 600 <= token < 700: return ("scale", (token - 600) / 10.0)
    if 700 <= token < 800: return ("angle", (token - 700) * 15)
    if 800 <= token < 900: 
        code = token - 800
        return ("flip", (code & 1, (code & 2) >> 1))
    return ("unknown", 0)

def top_k_sampling(logits, k=40, temp=0.8):
    # logits: 1D tensor of size [vocab_size]
    logits = logits / temp
    
    # Get top-k values and their indices
    values, indices = torch.topk(logits, k)  # both 1D, size=[k]
    
    # Softmax over top-k values
    probs = F.softmax(values, dim=-1)
    
    # Sample one index from top-k
    choice_idx = torch.multinomial(probs, 1)  # returns tensor([i])
    
    # Map back to original token
    choice_token = indices[choice_idx.item()] 
    return choice_token.item()


def generate():
    print("Loading model...")
    model = SkinTransformer().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(e)
        print("Train model first!")
        return
    
    model.eval()
    skins = []
    
    print(f"Generating {NUM_GEN} skins...")
    
    for i in tqdm(range(NUM_GEN), desc="Generating", unit="skin"):
        # Start with SOS
        curr_seq = [1]
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(250): # Max steps
                inp = torch.tensor([curr_seq], dtype=torch.long).to(DEVICE)
                logits = model(inp)
                last_logit = logits[0, -1, :]
                
                token = top_k_sampling(last_logit, k=30, temp=1.05)
                
                if token == 2: # EOS
                    break
                curr_seq.append(token)
        
        # Decode Tokens to JSON
        # Format was: [SOS, R, G, B, L1_ID, L1_X... ]
        # if len(curr_seq) < 5: continue
        
        # Parse Background Color
        # tokens 1, 2, 3 are R, G, B
        try:
            br = token_to_val(curr_seq[1])[1]
            bg = token_to_val(curr_seq[2])[1]
            bb = token_to_val(curr_seq[3])[1]
            bc_hex = (br << 16) + (bg << 8) + bb
            
            skin_obj = {"bc": bc_hex, "layers": []}
            
            # Parse Layers
            # Pattern: ID, X, Y, Scale, Angle, Flip, R, G, B
            # 9 tokens per layer
            layer_tokens = curr_seq[4:]
            chunk_size = 9
            
            for j in range(0, len(layer_tokens), chunk_size):
                chunk = layer_tokens[j:j+chunk_size]
                if len(chunk) < 9: break
                
                # Check if first token is valid ID
                t_type, t_val = token_to_val(chunk[0])
                if t_type != "id": continue 
                
                layer = {
                    "id": t_val,
                    "x": token_to_val(chunk[1])[1],
                    "y": token_to_val(chunk[2])[1],
                    "scale": token_to_val(chunk[3])[1],
                    "angle": token_to_val(chunk[4])[1],
                    "flipX": bool(token_to_val(chunk[5])[1][0]),
                    "flipY": bool(token_to_val(chunk[5])[1][1]),
                    "color": (token_to_val(chunk[6])[1] << 16) + 
                             (token_to_val(chunk[7])[1] << 8) + 
                             token_to_val(chunk[8])[1]
                }
                skin_obj["layers"].append(layer)
            
            skins.append(skin_obj)
            
        except Exception as e:
            print(f"Error parsing skin {i}: {e}")
            continue

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(skins[0], f, indent=2)
    print("Done.")

if __name__ == "__main__":
    generate()