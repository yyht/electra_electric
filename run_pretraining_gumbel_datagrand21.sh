nohup python3 run_pretraining_gumbel.py \
	--data-dir gs://yyht_source/pretrain \
	--data-file-list datagrand21/datagrand_21_file_list.txt \
	--model-name chinese_datagrand21_self_critic_gumbel_st \
	--hparams electric_gumbel_datagrand.json \
	--generator-ckpt models/bert_tiny_50g/model.ckpt-1000000 \
	--discriminator-ckpt models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt
	--vocab_file ./vocab/vocab_cn_datagrand.txt