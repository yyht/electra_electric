nohup python3 run_pretraining_electra.py \
	--data-dir gs://yyht_source/pretrain \
	--data-file-list datagrand21/datagrand_21_file_list.txt \
	--model-name electra_bert_base_datagrand \
	--hparams electra_base_ori_params.json
