nohup python run_pretraining.py \
	--data-dir gs://yyht_source/pretrain \
	--data-file-list uncased_english_whole_sentence_v3_32/uncased_english_whole_sentence_file.txt \
	--model-name uncased_en_electric_pretrained_mlm_electra_nce_topk_20 \
	--hparams electric_nce_sent_span_params_pretrained_uncased_english_one_stage_merged.json \
	--generator-ckpt models/uncased_L-4_H-512_A-8/bert_model.ckpt \
	--discriminator-ckpt models/uncased_L-12_H-768_A-12/bert_model.ckpt
