from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import torch
from copy import deepcopy


@dataclass
class DataCollatorForTEL:
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, instances):
        input_ids = [instance["input_ids"][:8192] for instance in instances]
        labels = [instance["labels"][:8192] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        return dict(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
        )