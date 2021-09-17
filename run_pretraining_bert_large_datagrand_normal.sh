nohup python3 run_pretraining_bert.py \
	--bert_config_file ./config/bert_config_datagrand_large.json \
	--input_file datagrand21/datagrand_21_file_list.txt \
	--output_dir gs://yyht_source/pretrain/models/bert_large_datagrand \
	--input_data_dir gs://yyht_source/pretrain \
	--init_checkpoint models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt \
	--max_seq_length 512 \
	--do_train True \
	--train_batch_size 128 \
	--learning_rate 1e-4 \
	--num_train_steps 2000000 \
	--num_warmup_steps 10000 \
	--save_checkpoints_steps 10000 \
	--iterations_per_loop 1000 \
	--use_tpu True \
	--tpu_name albert3 \
	--num_tpu_cores 8 \
	--eval_batch_size 256 \
	--max_predictions_per_seq 78 \
	--monitoring True \
	--lr_decay_power 1.0 \
	--weight_decay_rate 0.01 \
	--mask_strategy "span_mask" \
	--model_fn_type "normal" \
	--kld_ratio 1.0 \
	--simcse_ratio 1.0
