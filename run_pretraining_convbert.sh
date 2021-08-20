nohup python3 run_pretraining_convbert.py \
	--bert_config_file ./config/bert_config_base_official_conv_datagrand.json \
	--input_file datagrand21/datagrand_21_file_list.txt \
	--output_dir gs://yyht_source/pretrain/models/datagrand_21_convbert_base_simcse \
	--input_data_dir gs://yyht_source/pretrain \
	--max_seq_length 512 \
	--init_checkpoint models/datagrand_21_convbert_base/model.ckpt-300000 \
	--do_train True \
	--train_batch_size 128 \
	--learning_rate 1e-4 \
	--num_train_steps 1000000 \
	--num_warmup_steps 10000 \
	--save_checkpoints_steps 10000 \
	--iterations_per_loop 1000 \
	--use_tpu True \
	--tpu_name albert4 \
	--num_tpu_cores 8 \
	--eval_batch_size 256 \
	--max_predictions_per_seq 78 \
	--monitoring True \
	--lr_decay_power 1.0 \
	--weight_decay_rate 0.01 \
	--mask_strategy "span_mask" \
	--if_simcse True \
	--model_fn_type 'rdropout' \
	--kld_ratio 1.0 \
	--simcse_ratio 1.0
