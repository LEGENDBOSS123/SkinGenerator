import torch
import torch.nn as nn
import math

class SkinTransformer(nn.Module):
    def __init__(self, 
                 vocab_size=804, # Large enough to cover all our bins (max is ~803)
                 d_model=256, 
                 nhead=4,
                 num_layers=6,
                 dropout=0.05,
                 max_len=160):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # x: [Batch, Seq_Len] (Indices)
        seq_len = x.size(1)
        
        # Positions: [0, 1, 2, ... seq_len-1]
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        emb = self.embedding(x) + self.pos_embedding(positions)
        
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
            
        out = self.transformer(emb, mask=mask, is_causal=True)
        out = self.layer_norm(out)
        logits = self.output_head(out)
        
        return logits

    def configure_optimizers(self, lr=3e-4):
        return torch.optim.AdamW(self.parameters(), lr=lr)