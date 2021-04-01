nohup python run_pretraining_roformer_nystromformer.py \
	--bert_config_file ./config/bert_config_base_roformer_nystromformer.json \
	--input_file chinese_simplified_whole_sentence_v3_32/chinese_simplified_whole_sentence_file.txt \
	--output_dir gs://yyht_source/pretrain/models/bert_base_roformer_nystromformer_conv_50g \
	--input_data_dir gs://yyht_source/pretrain \
	--max_seq_length 256 \
	--do_train True \
	--train_batch_size 512 \
	--learning_rate 1e-4 \
	--num_train_steps 1000000 \
	--num_warmup_steps 10000 \
	--save_checkpoints_steps 1000 \
	--iterations_per_loop 1000 \
	--use_tpu True \
	--tpu_name albert1 \
	--num_tpu_cores 8 \
	--eval_batch_size 256 \
	--max_predictions_per_seq 78 \
	--monitoring False \
	--lr_decay_power 1.0 \
	--weight_decay_rate 0.01 \
	--mask_strategy "span_mask"
