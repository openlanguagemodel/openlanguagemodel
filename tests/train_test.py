import sys, os, torch
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from olm.data.datasets.fineweb_edu import FineWebEduDataset
from olm.data.datasets import DataLoader
from olm.train.trainer import Trainer
from olm.nn.blocks import LM
from olm.train.optim import AdamW

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LM(vocab_size=50257, embed_dim=768, num_heads=12, num_layers=12, max_seq_len=2048, dropout=0.1, ff_multiplier=4.0) # OPT-125M configuration with FineWeb Edu dataset
optimizer = AdamW(model.parameters(), 3e-4)
dataset = FineWebEduDataset(split="train", context_length=512, subset="sample-10BT", streaming=True)
dataloader = DataLoader(dataset, batch_size=2, num_workers=0, pin_memory=True)
trainer = Trainer(model, optimizer, dataloader, device, 512, use_amp=False)
losses = trainer.train(1, 10, 100)
print(f"S:{losses[0]:.4f} E:{losses[-1]:.4f} OK:{losses[-1]<losses[0]}")