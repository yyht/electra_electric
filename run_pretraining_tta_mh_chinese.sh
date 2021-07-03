nohup python3 run_pretraining_tta_mh.py \
	--data-dir gs://yyht_source/pretrain \
	--data-file-list chinese_simplified_whole_sentence_v3_32/chinese_simplified_whole_sentence_file.txt \
	--model-name mlm_electra_energy_chinese_small_untied_embed \
	--hparams mlm_energy_nce_chinese_small.json \
	--generator-ckpt models/bert_tiny_50g/model.ckpt-1000000 \
	--discriminator-ckpt models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt
