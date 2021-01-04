nohup python run_pretraining.py \
	--data-dir gs://yyht_source/pretrain \
	--data-file-list chinese_simplified_whole_sentence_v3_32/chinese_simplified_whole_sentence_file.txt \
<<<<<<< HEAD
	--model-name electric_nce_base_tied_embed_disallow \
	--hparams electric_nce_sent_params.json
=======
	--model-name electric_nce_base_tied_embed_disallow_v1 \
	--hparams electric_nce_sent_params.json
>>>>>>> fc43e19f4050be7d49f7ae49c4bbbbadcb57f3b0
