import os
import sys

from transformers import HfArgumentParser

from src.arguments import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
)
from src.models import AutoRelationExtractionModel

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # print(model_args)
    # print(data_args)
    # print(training_args)

    # 1. create model
    model = AutoRelationExtractionModel(model_args=model_args, training_args=training_args)

    # 2. finetune model
    # model.finetune(data_args, num_sanity_val_steps=0, accelerator='cpu', use_distributed_sampler=False, limit_train_batches=0.001, limit_val_batches=0.01)
    # model.finetune(data_args, num_sanity_val_steps=0, limit_val_batches=0.01)
    # model.finetune(data_args, num_sanity_val_steps=0, limit_train_batches=0.05, limit_val_batches=0.01)
    model.finetune(data_args, num_sanity_val_steps=0)
    
    # model.finetune(data_args, num_sanity_val_steps=0, devices=[1])

    os.remove(os.path.join(training_args.output_dir, "best_model.ckpt"))


if __name__ == '__main__':
    main()
