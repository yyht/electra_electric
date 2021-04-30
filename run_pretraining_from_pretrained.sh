nohup python3 run_pretraining.py \
	--data-dir gs://yyht_source/pretrain \
	--data-file-list chinese_simplified_whole_sentence_v3_32/chinese_simplified_whole_sentence_file.txt \
	--model-name electric_pretrained_mlm_electra_nce_topk_20 \
	--hparams electric_nce_sent_span_params_pretrained_one_stage_merged.json \
	--generator-ckpt models/bert_tiny_50g/model.ckpt-1000000 \
	--discriminator-ckpt models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt
