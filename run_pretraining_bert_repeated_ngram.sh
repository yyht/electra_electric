nohup python3 ./run_pretraining_bert_repeated_ngram.py \
	--bert_config_file ./config/bert_config.json \
	--data_path_dict ./config/chinese_data_config.json \
	--output_dir gs://yyht_source/pretrain/models/roberta_base_repeated_ngram_sop_chinese \
	--input_data_dir gs://yyht_source/pretrain \
	--buckets gs://yyht_source/pretrain \
	--vocab_path ./vocab/vocab_ch_all.txt \
	--max_seq_length 512 \
	--do_train True \
	--train_batch_size 256 \
	--learning_rate 1e-4 \
	--num_train_steps 1000000 \
	--num_warmup_steps 10000 \
	--save_checkpoints_steps 1000 \
	--iterations_per_loop 1000 \
	--use_tpu True \
	--tpu_name albert3 \
	--num_tpu_cores 8 \
	--eval_batch_size 256 \
	--max_predictions_per_seq 78 \
	--lr_decay_power 1.0 \
	--weight_decay_rate 0.01 \
	--mask_ratio 0.15 \
	--random_ratio 0.1 \
	--min_tok 3 \
	--max_tok 10 \
	--mask_id 103 \
	--cls_id 101 \
	--sep_id 102 \
	--pad_id 0 \
	--geometric_p 0.1 \
	--max_pair_targets 10 \
	--break_mode 'doc' \
	--doc_stride 64 \
	--doc_num 5
