import torch

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)

tf_checkpoint_path = '/Users/xuhaotian/Downloads/bert_large_50g_ilm_final/bert_large_50g_ilm_final_mixture_final_v1_model.ckpt-1000000'
bert_config_file = '/Users/xuhaotian/Downloads/bert_large_50g_ilm_final/bert_config_ilm_large.json'
pytorch_dump_path = '/Users/xuhaotian/Downloads/bert_large_50g_ilm_final/bert_large_ilm_final_mixture_final_v1.pth'

convert_tf_checkpoint_to_pytorch(tf_checkpoint_path=tf_checkpoint_path,
                                bert_config_file=bert_config_file,
                                pytorch_dump_path=pytorch_dump_path)