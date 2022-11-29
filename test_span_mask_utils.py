from pretrain import span_mask_utils_ilm
from pretrain import pretrain_helpers

from bunch import Bunch
import numpy as np
import tensorflow as tf


FLAGS = Bunch({})
FLAGS.min_tok = 3
FLAGS.max_tok = 5
FLAGS.sep_id = 102
FLAGS.pad_id = 0
FLAGS.cls_id = 101
FLAGS.mask_id = 103
FLAGS.batch_size = 2
FLAGS.leak_ratio = 0.1
FLAGS.rand_ratio = 0.1
FLAGS.vocab_size = 21128
FLAGS.mask_prob = 0.01
FLAGS.sample_strategy = 'token_span'
FLAGS.confusion_matrix = None
FLAGS.confusion_mask_matrix = None
FLAGS.prepare_text_infilling = False
FLAGS.initial_ratio = 0.1
FLAGS.final_ratio = 0.1
FLAGS.num_train_steps = 1000
FLAGS.seg_id = 105 # <S>
FLAGS.ilm_v1 = False
FLAGS.ilm_v2 = True

name_to_features = {
      "input_ori_ids":
        tf.FixedLenFeature([512], tf.int64)
}
record_spec = {
        "input_ori_ids":
            tf.FixedLenFeature([512], tf.int64),
#         "input_mask":
#             tf.FixedLenFeature([512], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([512], tf.int64),
  }

def _decode_record(record, name_to_features, **kargs):
  example = tf.parse_single_example(record, name_to_features)
  return example

def train_input_fn(input_file, _parse_fn, name_to_features,
           params,
           num_predict=78, seq_len=512,
           use_bfloat16=False,
           truncate_seq=False,
           stride=1,
          **kargs):

  dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))
#     dataset = dataset.shuffle(1024)
#     dataset = dataset.map(lambda x:_parse_fn(x, record_spec))
  dataset = dataset.map(lambda x:span_mask_utils_ilm._decode_record(FLAGS, x, num_predict,
                  seq_len, 
                  use_bfloat16=use_bfloat16, 
                  truncate_seq=truncate_seq, 
                  stride=stride))
  dataset = dataset.batch(params.get("batch_size", 1))
  dataset = dataset.repeat(params.get("epoch", 1))
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  return features

output = ['/Users/xuhaotian/Downloads/chinese_sub_task_45.tfrecord']
input_fn = train_input_fn(output[0], _decode_record, name_to_features, params=FLAGS)

sess = tf.Session()

init_op = tf.group(
      tf.local_variables_initializer())
sess.run(init_op)
cout = 0
while True:
  features = sess.run(input_fn)
  cout += 1
  break

from tokenizers import (ByteLevelBPETokenizer,
      CharBPETokenizer,
      SentencePieceBPETokenizer,
      BertWordPieceTokenizer)

vocab = './vocab/vocab_ch.txt'

chinese_bpe_tokenizer = BertWordPieceTokenizer(
    vocab, 
    lowercase=True)

print(chinese_bpe_tokenizer.decode(features['ilm_input'][0], skip_special_tokens=False))
print(chinese_bpe_tokenizer.decode(features['origin_input'][0], skip_special_tokens=False))

print(features['ilm_segment_ids'][0])
