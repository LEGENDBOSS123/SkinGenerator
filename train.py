import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import SkinTransformer
import time
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 50 # Might need more epochs for discrete tokens
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def train():
    # Load Data
    try:
        data = torch.load('skins_tokens.pt')
    except:
        print("Run preprocess.py first!")
        return

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Init Model
    model = SkinTransformer().to(DEVICE)
    optimizer = model.configure_optimizers(lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding
    
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in progress:
            # batch is a list [tensor], so take batch[0]
            inputs = batch[0].to(DEVICE) # [Batch, Seq]
            
            # Shift targets: Input [A, B, C], Target [B, C, D]
            input_seq = inputs[:, :-1]
            target_seq = inputs[:, 1:]
            
            logits = model(input_seq) # [Batch, Seq-1, Vocab]
            
            # Flatten for loss
            loss = criterion(logits.reshape(-1, 804), target_seq.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            progress.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")
        
        # if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), f"checkpoints2/skin_model_v2_ep{epoch+1}.pth")

if __name__ == "__main__":
    train()