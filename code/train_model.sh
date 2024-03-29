python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--model_name_or_path "klue/bert-base" \
--eval_steps 500 \
--logging_steps 500 \
--evaluation_strategy "steps" \
# --per_device_train_batch_size 8 \
# --per_device_eval_batch_size 8 \
# --num_train_epochs 3 \
# --weight_decay 0.01 \
# --warmup_ratio 0.1 \
# --learning_rate 3e-5 \
# --load_best_model_at_end True \ 
# --metric_for_best_model "exact_match"