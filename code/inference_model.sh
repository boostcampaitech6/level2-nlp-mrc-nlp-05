python inference.py \
--output_dir "./outputs/test_dataset/" \
--dataset_name "../data/test_dataset/" \
--valid_dataset_name "../data/train_dataset/" \
--model_name_or_path "./models/train_dataset/" \
--retrieval_class "bm25" \
--bm25_type "bm25L" \
--do_predict