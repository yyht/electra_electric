nohup python run_pretraining.py \
	--data-dir gs://yyht_source/pretrain \
	--data-file-list chinese_simplified_whole_sentence_v3_32/chinese_simplified_whole_sentence_file.txt \
	--model-name electric_nce_span_mask_from_pretrained_nce_new \
	--hparams electric_nce_sent_span_params_pretrained.json \
	--generator-ckpt models/bert_tiny_50g/model.ckpt-1000000 \
	--discriminator-ckpt models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt
