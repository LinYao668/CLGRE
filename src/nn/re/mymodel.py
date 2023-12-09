import itertools
from collections import Counter
from typing import Optional, List, Any

import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..model_utils import RelationExtractionOutput, MODEL_MAP
from ...datasets.utils import tensor_to_numpy
from ...layers.global_pointer import EfficientGlobalPointer
from ...losses import SparseMultilabelCategoricalCrossentropy


def get_auto_mymodel_re_model(
    model_type: Optional[str] = "bert",
    base_model: Optional[PreTrainedModel] = None,
    parent_model: Optional[PreTrainedModel] = None,
):
    print('Model:')
    if base_model is None and parent_model is None:
        base_model, parent_model = MODEL_MAP[model_type]

    class MyModel(parent_model):
        """
        基于`BERT`的`MyModel`关系抽取模型
        + 📖 模型的整体思路将三元组抽取分解为实体首尾对应、主体-客体首首对应、主体-客体尾尾对应
        + 📖 通过采用类似多头注意力得分计算的机制将上述三种关系最后映射到一个二维矩阵
        + 📖 每种关系都采用`GlobalPointer`来建模

        Args:
            `config`: 模型的配置
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            setattr(self, self.base_model_prefix, base_model(config, add_pooling_layer=False))

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.hidden_size = config.hidden_size
            
            self.entity_tagger = EfficientGlobalPointer(config.hidden_size, 2, config.head_size)
            self.head_tail_tagger = EfficientGlobalPointer(config.hidden_size, 2, config.head_size, use_rope=False, tril_mask=False)
            self.tail_head_tagger = EfficientGlobalPointer(config.hidden_size, 2, config.head_size, use_rope=False, tril_mask=False)
            
            
            self.head_rel_tagger = EfficientGlobalPointer(config.hidden_size, 2, config.head_size)
            self.tail_rel_tagger = EfficientGlobalPointer(config.hidden_size, 2, config.head_size)
            
            
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            entity_labels: Optional[torch.Tensor] = None,
            head_labels: Optional[torch.Tensor] = None,
            tail_labels: Optional[torch.Tensor] = None,
            num_tokens_relations: Optional[int] = None,
            relation_index: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            offset_mapping: Optional[List[Any]] = None,
            target: Optional[List[Any]] = None,
        ) -> RelationExtractionOutput:

            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            sequence_output = self.dropout(outputs[0])  # [batch_size, seq_len + num_predicates, hidden_size]
            relations_sequence_output = torch.index_select(sequence_output[:, -num_tokens_relations:, :], 1, relation_index.to(sequence_output.device))
            
            num_relations = len(relation_index)
            entity_logits = self.entity_tagger(sequence_output[:, :-num_tokens_relations, :], mask=attention_mask[:, :-num_tokens_relations])
            sequence_with_relation = torch.cat([sequence_output[:, :-num_tokens_relations, :], relations_sequence_output], dim=1)
            head_tail_matrix = self.head_tail_tagger(sequence_with_relation, mask=attention_mask[:, :sequence_with_relation.shape[1]])
            tail_head_matrix = self.tail_head_tagger(sequence_with_relation, mask=attention_mask[:, :sequence_with_relation.shape[1]])
            
            head_tail = head_tail_matrix[:, :, :-num_relations, :-num_relations]
            head_tail = head_tail.sum(dim=1, keepdim=True)
            
            tail_head = tail_head_matrix[:, :, :-num_relations, :-num_relations]
            tail_head = tail_head.sum(dim=1, keepdim=True)
            
            entity_logits = torch.cat([entity_logits, tail_head, head_tail], dim=1)
            
            _head_logits = head_tail_matrix[:, :, :-num_relations, -num_relations:]
            _tail_logits = head_tail_matrix[:, :, -num_relations:, :-num_relations].transpose(3, 2)
            
            head_logits = self.head_rel_tagger(sequence_output, mask=attention_mask)[:, :, :-num_tokens_relations, -num_tokens_relations:]
            tail_logits = self.tail_rel_tagger(sequence_output, mask=attention_mask)[:, :, :-num_tokens_relations, -num_tokens_relations:]
            head_logits = torch.index_select(head_logits, 3, relation_index.to(head_logits.device))
            tail_logits = torch.index_select(tail_logits, 3, relation_index.to(head_logits.device))

            loss, predictions = None, None
            if entity_labels is not None and head_labels is not None and tail_labels is not None:
                entity_loss = self.compute_loss([entity_logits, entity_labels])
                head_loss = self.compute_loss([head_logits, head_labels])
                tail_loss = self.compute_loss([tail_logits, tail_labels])
                _head_loss = self.compute_loss([_head_logits, head_labels])
                _tail_loss = self.compute_loss([_tail_logits, tail_labels])
                
                combine_head_loss = self.compute_loss([head_logits + _head_logits, head_labels])
                combing_tail_loss = self.compute_loss([tail_logits + _tail_logits, tail_labels])
                loss = (entity_loss + combine_head_loss + combing_tail_loss + (head_loss + tail_loss) / 2 + (_head_loss + _tail_loss) / 2) / 5
            if not self.training:
                predictions = self.decode(
                    entity_logits, head_logits + _head_logits, tail_logits + _tail_logits, attention_mask, texts, offset_mapping)

            return RelationExtractionOutput(
                loss=loss,
                logits=None,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, entity_logits, head_logits, tail_logits, masks, texts, offset_mapping):
            all_spo_list = []
            shape = entity_logits.shape
            batch_size, sentence_length = shape[0], shape[2]
            masks = tensor_to_numpy(masks)[:, :sentence_length]

            entity_logits = tensor_to_numpy(entity_logits)
            head_logits = tensor_to_numpy(head_logits)
            tail_logits = tensor_to_numpy(tail_logits)
            decode_thresh = getattr(self.config, "decode_thresh", 0.0)

            id2predicate = {int(v): k for k, v in self.config.predicate2id.items()}
            for bs in range(batch_size):
                subjects, objects, sub_tail_obj_head, sub_head_obj_tail = set(), set(), set(), set()
                subjects_head, objects_head = set(), set()
                subjects_tail, objects_tail = set(), set()
                sub_obj_head, sub_tail_obj = set(), set()
                sub_head_obj, sub_obj_tail = set(), set()
                _entity_logits = entity_logits[bs]
                
                l = masks[bs].sum()
                text, mapping = texts[bs], offset_mapping[bs]
                for r, h, t in zip(*np.where(_entity_logits > decode_thresh)):
                    if h >= (l - 1) or t >= (l - 1) or 0 in [h, t]:  # 排除[CLS]、[SEP]、[PAD]
                        continue
                    if r == 0:
                        subjects.add((h, t))
                        subjects_head.add(h)
                        subjects_tail.add(t)
                    elif r == 1:
                        objects.add((h, t))
                        objects_head.add(h)
                        objects_tail.add(t)
                    elif r == 2:
                        sub_tail_obj_head.add((h, t))
                        sub_tail_obj.add(h)
                        sub_obj_head.add(t)
                    else: 
                        sub_head_obj_tail.add((h, t))
                        sub_head_obj.add(h)
                        sub_obj_tail.add(t)
                
                sub_obj = set()
                for (sh, ot), (st, oh) in itertools.product(sub_head_obj_tail, sub_tail_obj_head):
                    
                    if (sh, st) in subjects and (oh, ot) in objects and sh != st and oh != ot:
                        sub_obj.add(((sh, st), (oh, ot)))
                sub_h_rel, sub_t_rel, obj_h_rel, obj_t_rel = {}, {}, {}, {}
                _head_logits, _tail_logits = head_logits[bs], tail_logits[bs]
                for r, e, p in zip(*np.where(_head_logits > decode_thresh)):
                    if r == 0:
                        if e not in sub_h_rel:
                            sub_h_rel[e] = set({p})
                        sub_h_rel[e].add(p)
                    else: 
                        if e not in sub_t_rel:
                            sub_t_rel[e] = set({p})
                        sub_t_rel[e].add(p)
                
                for r, e, p in zip(*np.where(_tail_logits > decode_thresh)):
                    if r == 0:
                        if e not in obj_h_rel:
                            obj_h_rel[e] = set({p})
                        obj_h_rel[e].add(p)
                    else: 
                        if e not in obj_t_rel:
                            obj_t_rel[e] = set({p})
                        obj_t_rel[e].add(p)
                
                spoes = set()
                for (sh, st), (oh, ot) in sub_obj:
                    
                    sub_head_rel = sub_h_rel.get(sh, set())
                    sub_tail_rel = sub_t_rel.get(st, set())
                    
                    rel_in_sub = sub_head_rel | sub_tail_rel
                    
                    obj_head_rel = obj_h_rel.get(oh, set())
                    obj_tail_rel = obj_t_rel.get(ot, set())
                    
                    rel_in_obj = obj_head_rel | obj_tail_rel
                    
                    ps = rel_in_sub & rel_in_obj
                    
                    ps = list(ps)
                    if len(ps) > 0:
                        for p in ps: 
                            spoes.add((
                                id2predicate[p],
                                text[mapping[sh][0]:mapping[st][1]],
                                text[mapping[oh][0]:mapping[ot][1]]
                            ))

                all_spo_list.append(spoes)
            return all_spo_list

        def compute_loss(self, inputs):
            preds, target = inputs[:2]
            shape = preds.shape
            target = target[..., 0] * shape[3] + target[..., 1]  # [bsz, heads, num_spoes]
            # np.prod 是累积
            preds = preds.reshape(shape[0], -1, np.prod(shape[2:]))
            loss_fct = SparseMultilabelCategoricalCrossentropy(mask_zero=True)
            loss = loss_fct(preds, target.long()).sum(dim=1).mean()
            return loss

    return MyModel


def get_mymodel_re_model_config(predicates, **kwargs):
    predicate2id = {v: i for i, v in enumerate(predicates)}
    model_config = {
        "num_predicates": len(predicates), "predicate2id": predicate2id, "head_size": 64,
    }
    model_config.update(kwargs)
    return model_config
