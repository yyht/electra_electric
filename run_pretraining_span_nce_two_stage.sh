nohup python run_pretraining.py \
	--data-dir gs://yyht_source/pretrain \
	--data-file-list chinese_simplified_whole_sentence_v3_32/chinese_simplified_whole_sentence_file.txt \
	--model-name electric_nce_base_tied_embed_disallow_v5_span_nce_two_stage \
	--hparams electric_nce_sent_span_params_two_stage.json
