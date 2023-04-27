import lightning.pytorch as pl
from src.model_utils import LMAFT as LM
from torch.nn import Linear, ModuleList
from torch.nn import functional as F
from torch.optim import Adam


class LightningAFT(pl.LightningModule):
    def __init__(self, vocab_size, block_size, embedding_dim, layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.model = LM(vocab_size, block_size, embedding_dim, layers)
        self.save_hyperparameters()

    def forward(self, idx):
        return self.model.forward(idx)

    def training_step(self, batch, batch_idx):
        idx, labels = batch
        idx = idx[:, :self.block_size]
        labels = labels[:, :self.block_size]
        logits = self.forward(idx)
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        idx, labels = batch
        idx = idx[:, :self.block_size]
        labels = labels[:, :self.block_size]
        logits = self.forward(idx)
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)