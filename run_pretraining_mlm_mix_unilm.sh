nohup python3 run_pretraining_mlm_mix_unilm.py \
	--bert_config_file ./config/bert_config_ilm.json \
	--input_file chinese_simplified_whole_sentence_v3_32/chinese_simplified_whole_sentence_file.txt \
	--output_dir gs://yyht_source/pretrain/models/bert_base_50g_mix_ilm_final_v1 \
	--input_data_dir gs://yyht_source/pretrain \
	--init_checkpoint models/chinese_L-12_H-768_A-12_ilm_v1/bert_model.ckpt \
	--max_seq_length 512 \
	--do_train True \
	--train_batch_size 128 \
	--learning_rate 1e-4 \
	--num_train_steps 1000000 \
	--num_warmup_steps 10000 \
	--save_checkpoints_steps 10000 \
	--iterations_per_loop 1000 \
	--use_tpu True \
	--tpu_name albert2 \
	--num_tpu_cores 8 \
	--eval_batch_size 256 \
	--max_predictions_per_seq 76 \
	--monitoring False \
	--lr_decay_power 1.0 \
	--weight_decay_rate 0.01 \
	--mask_strategy "span_mask"
