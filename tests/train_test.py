import sys, os, torch, urllib.request; from torch.utils.data import DataLoader; from tempfile import TemporaryDirectory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from olm.data.datasets import Dataset; from olm.data.tokenization.hf_tokenizer import HFTokenizer; from olm.train.trainer import Trainer; from olm.nn.blocks import LM


with TemporaryDirectory() as tmp:
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", os.path.join(tmp, "i.txt"))
    tokenizer, device = HFTokenizer("gpt2"), "cuda" if torch.cuda.is_available() else "cpu"
    model = LM(tokenizer.vocab_size, 64, 4, 2, 33)
    optimizer = torch.optim.AdamW(model.parameters(), 3e-4)
    dataset = Dataset(tmp, tokenizer, 32)
    dataloader = DataLoader(dataset, 4)
    trainer = Trainer(model, optimizer, dataloader, device, 32, use_amp=False)
    losses = trainer.train(1, 10, 100)
    print(f"S:{losses[0]:.4f} E:{losses[-1]:.4f} OK:{losses[-1]<losses[0]}")
