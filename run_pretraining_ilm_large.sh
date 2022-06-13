nohup python3 run_pretraining_ilm.py \
	--bert_config_file ./config/bert_config_ilm_large.json \
	--input_file chinese_simplified_whole_sentence_v3_32/chinese_simplified_whole_sentence_file.txt \
	--output_dir gs://yyht_source/pretrain/models/bert_large_50g_ilm_final \
	--input_data_dir gs://yyht_source/pretrain \
	--init_checkpoint models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16_ilm/bert_model.ckpt.ckpt \
	--max_seq_length 512 \
	--do_train True \
	--train_batch_size 64 \
	--learning_rate 1e-4 \
	--num_train_steps 1000000 \
	--num_warmup_steps 10000 \
	--save_checkpoints_steps 10000 \
	--iterations_per_loop 1000 \
	--use_tpu True \
	--tpu_name albert0 \
	--num_tpu_cores 8 \
	--eval_batch_size 256 \
	--max_predictions_per_seq 128 \
	--monitoring True \
	--lr_decay_power 1.0 \
	--weight_decay_rate 0.01 \
	--mask_strategy "span_mask"