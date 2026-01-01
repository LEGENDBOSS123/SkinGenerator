import json
import torch
from tqdm import tqdm

# --- CONFIG ---
INPUT_FILE = 'avatars.json'
OUTPUT_FILE = 'skins_tokens.pt'

# --- VOCABULARY MAPPING ---
# We map every property to a unique integer range.
# Special: 0=PAD, 1=SOS, 2=EOS
# Layer IDs (1-115): Map to 100-215
# X (-32 to 32): Map to 300-364
# Y (-32 to 32): Map to 400-464
# Colors (0-255): Quantize to 32 bins -> Map to 500-531
# Scale (0-2.0): Quantize to 20 bins -> Map to 600-620
# Angle (0-360): Quantize to 24 bins -> Map to 700-724
# Flip (0-3): Map to 800-803

def get_token_id(val): return 100 + val
def get_token_x(val): return 300 + int(max(-32, min(32, val)) + 32)
def get_token_y(val): return 400 + int(max(-32, min(32, val)) + 32)
def get_token_color(val): return 500 + int((val / 255.0) * 31)
def get_token_scale(val): return 600 + int(min(2.0, max(0, val)) * 10)
def get_token_angle(val): return 700 + int((val % 360) / 15)
def get_token_flip(fx, fy): 
    # 0: None, 1: X, 2: Y, 3: Both
    code = (1 if fx else 0) + (2 if fy else 0)
    return 800 + code

def hex_to_rgb(hex_val):
    hex_val = int(hex_val)
    r = (hex_val >> 16) & 255
    g = (hex_val >> 8) & 255
    b = hex_val & 255
    return r, g, b

def process_data():
    print(f"Loading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return

    processed_tensors = []
    max_len = 0
    
    for skin in tqdm(data, desc="Preprocessing", unit="skin"):
        # Sequence structure: 
        # [SOS, BG_R, BG_G, BG_B, L1_ID, L1_X, L1_Y, L1_Scale, L1_Ang, L1_Flip, L1_R, L1_G, L1_B, ..., EOS]
        
        seq = [1] # SOS
        
        # Background Color
        br, bg, bb = hex_to_rgb(skin.get('bc', 0))
        seq.extend([get_token_color(br), get_token_color(bg), get_token_color(bb)])
        
        layers = skin.get('layers', [])
        # Optional: Sort layers by ID to help model learn structure? 
        # layers.sort(key=lambda x: x.get('id', 0))
        
        for layer in layers:
            lid = layer.get('id', 1)
            lx = layer.get('x', 0)
            ly = layer.get('y', 0)
            ls = layer.get('scale', 1)
            la = layer.get('angle', 0)
            fx = layer.get('flipX', False)
            fy = layer.get('flipY', False)
            lr, lg, lb = hex_to_rgb(layer.get('color', 0))
            
            # Append tokens for this layer
            seq.append(get_token_id(lid))
            seq.append(get_token_x(lx))
            seq.append(get_token_y(ly))
            seq.append(get_token_scale(ls))
            seq.append(get_token_angle(la))
            seq.append(get_token_flip(fx, fy))
            seq.append(get_token_color(lr))
            seq.append(get_token_color(lg))
            seq.append(get_token_color(lb))
            
        seq.append(2) # EOS
        
        if len(seq) > 256: # Truncate if too long
            seq = seq[:256]
            
        processed_tensors.append(torch.tensor(seq, dtype=torch.long))
        max_len = max(max_len, len(seq))

    print(f"Max sequence length found: {max_len}")
    
    # Pad all to max_len
    padded_tensors = torch.zeros(len(processed_tensors), max_len, dtype=torch.long) # 0 is PAD
    for i, seq in enumerate(processed_tensors):
        padded_tensors[i, :len(seq)] = seq
        
    print(f"Saving tensor shape {padded_tensors.shape} to {OUTPUT_FILE}...")
    torch.save(padded_tensors, OUTPUT_FILE)

if __name__ == "__main__":
    process_data()