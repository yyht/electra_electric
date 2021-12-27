nohup python3 run_pretraining_mlm_mix_unilm.py \
	--bert_config_file ./config/bert_config_ilm_uncased_en.json \
	--input_file uncased_english_whole_sentence_v3_32/uncased_english_whole_sentence_file.txt \
	--output_dir gs://yyht_source/pretrain/models/bert_base_50g_mix_ilm_uncased_en_final \
	--input_data_dir gs://yyht_source/pretrain \
<<<<<<< HEAD
=======
	--init_checkpoint models/uncased_L-12_H-768_A-12_ilm_v1/bert_model.ckpt \
>>>>>>> 68b131c0e839473a33500c7c309986bffac58384
	--max_seq_length 512 \
	--do_train True \
	--init_checkpoint models/uncased_L-12_H-768_A-12_ilm/bert_model.ckpt \
	--train_batch_size 128 \
	--learning_rate 1e-4 \
	--num_train_steps 1000000 \
	--num_warmup_steps 10000 \
	--save_checkpoints_steps 10000 \
	--iterations_per_loop 1000 \
	--use_tpu True \
	--tpu_name albert1 \
	--num_tpu_cores 8 \
	--eval_batch_size 256 \
	--max_predictions_per_seq 72 \
	--monitoring False \
	--lr_decay_power 1.0 \
	--weight_decay_rate 0.01 \
	--mask_strategy "span_mask"
