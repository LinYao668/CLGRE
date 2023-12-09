from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict

import sys

import numpy as np

import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from ..iterable import IterableDataLoader

from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from .base import RelationExtractionDataModule
from ..utils import batchify_re_labels


@dataclass
class DataCollatorForMyModel:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None
    ignore_list: Optional[List[str]] = None
    predicates: Optional[List[str]] = None
    
    length_relation_sentence: Optional[int] = None
    relation_token_index: Optional[torch.Tensor] = None
    
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 num_predicates: Optional[int] = None,
                 ignore_list: Optional[List[str]] = None,
                 predicates: Optional[List[str]] = None,
                 length_relation_sentence: Optional[int] = None,
                 relation_token_index: Optional[torch.Tensor] = None,
                 flag: Optional[str] = None,
                 ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.num_predicates = num_predicates
        self.ignore_list = ignore_list
        self.predicates = predicates
        self.length_relation_sentence = length_relation_sentence
        self.relation_token_index = relation_token_index
        
        self.flag = flag
        
        self.relation_sentence_feature = self.set_relation_labels()
    
    def set_relation_labels(self) -> None:
        relation_token_ids = [self.tokenizer.encode(relation, add_special_tokens=False, return_tensors='np') for relation in self.predicates]
        _relation_token_ids = relation_token_ids[0][0]
        for x in relation_token_ids[1:]:
            _relation_token_ids = np.concatenate((_relation_token_ids, x[0]), axis=0)
        
        _relation_token_ids = torch.LongTensor(_relation_token_ids)
        
        relation_token_index = [0]
        for i in [len(x[0]) for x in relation_token_ids][:-1]:
            relation_token_index.append(relation_token_index[-1] + i)
        self.relation_token_index = torch.LongTensor(relation_token_index)
        self.length_relation_sentence = len(_relation_token_ids)

        return {
            'input_ids': _relation_token_ids,
            'attention_mask': torch.ones(self.length_relation_sentence).long(),
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in self.ignore_list} for f in features]
        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        bs = batch["input_ids"].size(0)
        
        batch_relation_input_ids = self.relation_sentence_feature['input_ids'].unsqueeze(0).repeat(bs, 1)
        batch_relation_attention_mask = self.relation_sentence_feature['attention_mask'].unsqueeze(0).repeat(bs, 1)
        
        batch['input_ids'] = torch.cat([batch['input_ids'], batch_relation_input_ids], dim=1) 
        batch['attention_mask'] = torch.cat([batch['attention_mask'], batch_relation_attention_mask], dim=1)

        if labels is None:  # for test
            batch['num_tokens_relations'] = self.length_relation_sentence
            batch['relation_index'] = self.relation_token_index
            return batchify_re_labels(batch, features, return_offset_mapping=True)

        max_spo_num = max(len(lb) for lb in labels)
        batch_entity_labels = torch.zeros(bs, 4, max_spo_num, 2, dtype=torch.long)
        batch_head_labels = torch.zeros(bs, 2, max_spo_num, 2, dtype=torch.long)
        batch_tail_labels = torch.zeros(bs, 2, max_spo_num, 2, dtype=torch.long)

        for i, lb in enumerate(labels):
            # (s, p, o): (subject, predicate, object)
            # (sh, st, p, oh, ot): 
            # sh 表示subject的首，st表示subject的尾
            # oh 表示object的首，ot表示object的尾
            # p 表示关系
            # print(lb)
            for spidx, (sh, st, p, oh, ot) in enumerate(lb):
                # sh * batch_max_length + st   < batch_max_length * batch_max_length
                batch_entity_labels[i, 0, spidx, :] = torch.tensor([sh, st])        # 表示subject的首尾
                batch_entity_labels[i, 1, spidx, :] = torch.tensor([oh, ot])        # 表示object的首尾
                batch_entity_labels[i, 2, spidx, :] = torch.tensor([st, oh])        
                batch_entity_labels[i, 3, spidx, :] = torch.tensor([sh, ot])
                
                
                # p * batch_max_length + sh   < num_relation * batch_max_length
                batch_head_labels[i, 0, spidx, :] = torch.tensor([sh, p])          # 表示subject的首-->关系    p-1 是由于计算机中下标的其实点是从0开始的。
                batch_head_labels[i, 1, spidx, :] = torch.tensor([st, p])          # 表示subject的尾-->关系
                
                # if (sh * batch_max_length + p - 1) > batch_max_length * num_relations
                batch_tail_labels[i, 0, spidx, :] = torch.tensor([oh, p])          # 表示object的首-->关系
                batch_tail_labels[i, 1, spidx, :] = torch.tensor([ot, p])          # 表示object的尾-->关系

        batch["entity_labels"] = batch_entity_labels
        batch["head_labels"] = batch_head_labels
        batch["tail_labels"] = batch_tail_labels
        
        batch['num_tokens_relations'] = self.length_relation_sentence
        batch['relation_index'] = self.relation_token_index
        

        return batch


class MyModelForReDataModule(RelationExtractionDataModule):

    config_name: str = "mymodel"
    
    relation_token_index: torch.Tensor = None
    length_relation_sentence: int = None

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        data_collator = DataCollatorForMyModel(
            tokenizer=self.tokenizer,
            num_predicates=len(self.labels),
            ignore_list=ignore_list,
            predicates=self.labels,
            flag='train',
        )
        
        return data_collator
        
        
    def val_dataloader(self) -> DataLoader:
        print('val:', self.streaming)
        
        ignore_list = ["offset_mapping", "text", "target"]
        data_collator = DataCollatorForMyModel(
            tokenizer=self.tokenizer,
            num_predicates=len(self.labels),
            ignore_list=ignore_list,
            predicates=self.labels,
            flag='val',
        )
        
        if self.streaming:
            return IterableDataLoader(
                self.ds["validation"],
                batch_size=self.validation_batch_size,
                num_workers=self.num_workers,
                collate_fn=data_collator,
            )

        return DataLoader(
            self.ds["validation"],
            batch_size=self.validation_batch_size,
            sampler=SequentialSampler(self.ds["validation"]),
            num_workers=self.num_workers,
            collate_fn=data_collator,
        )
