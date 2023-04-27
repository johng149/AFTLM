import torch
from torch.nn.utils.rnn import pad_sequence as pad

class CustomCollator:

    def __init__(self, tokenizer, data_col, pad_token_id=3):
        """
        Assumes tokenizer has a `encode_as_ids` method
        which takes a string and returns a list of integers

        Assumes data_col is a string where any features
        passed into the collator will have a key with that name
        which contains a string value
        """
        self.tokenizer = tokenizer
        self.data_col = data_col
        self.pad_id = pad_token_id

    def process_one(self, f):
        """
        Assumes f is a dictionary with a single key, `data_col`,
        which contains a string
        """
        input_ids = self.tokenizer.encode_as_ids(f[self.data_col])
        return torch.tensor(input_ids)
    
    def __call__(self, features):
        """
        Assumes features is a list of dictionaries
        """
        # flip(0) is needed for padding to work
        processed = [self.process_one(f).flip(0) for f in features]

        # now to pad all the sequences to the same length
        ids = pad(processed, batch_first=True, padding_value=self.pad_id).flip(1)
        labels = ids.clone()

        # replace all pad tokens with -100
        labels[labels == self.pad_id] = -100

        # all tokens save for last is input while
        # all tokens save for first is labels
        return ids, labels