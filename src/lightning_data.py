import lightning.pytorch as pl
from src.dataset import DatasetBuilder
from src.collator import CustomCollator
from torch.utils.data import DataLoader


class TextDataModule(pl.LightningDataModule):
    def __init__(self, data_col, batch_size, tokenizer, take=None):
        super().__init__()
        self.data_col = data_col
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.take = take
        self.builder = DatasetBuilder(data_col)
        self.train_collator = CustomCollator(tokenizer, data_col)
        self.val_collator = CustomCollator(tokenizer, data_col)

    def setup(self, stage=None):
        self.train, self.val = self.builder.auto(take=self.take)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.train_collator)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.val_collator)