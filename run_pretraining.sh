nohup python run_pretraining.py \
	--data-dir gs://yyht_source/pretrain \
	--data-file-list chinese_simplified_whole_sentence_v1/chinese_simplified_whole_sentence_file.txt \
	--model-name electric_base \
	--hparams electric_params.json