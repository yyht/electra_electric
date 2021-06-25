nohup python3 run_pretraining_tta_mh.py \
	--data-dir gs://yyht_source/pretrain \
	--data-file-list uncased_english_whole_sentence_v3_32/uncased_english_whole_sentence_file.txt \
	--model-name mlm_electra_energy_uncased_en_small \
	--hparams mlm_energy_nce_uncased_en_small.json \
	--generator-ckpt models/uncased_L-4_H-512_A-8/bert_model.ckpt \
	--discriminator-ckpt models/uncased_L-12_H-768_A-12/bert_model.ckpt
