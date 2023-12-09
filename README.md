

# 基于 Bert 的电力通讯法律文件关系抽取

## 实验复现

python train.py --task_name mymodel --model_name_or_path hfl/chinese-roberta-wwm-ext --model_type bert --dataset_name datasets/bre --train_file train.json --validation_file test.json --test_file test.json --cache_dir datasets/bre/mymodel --preprocessing_num_workers 16 --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --other_learning_rate 1e-4 --output_dir outputs/mymodel








