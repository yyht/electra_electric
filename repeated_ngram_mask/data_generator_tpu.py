
from tokenizer import data_generator
from tokenizer import sop_generator
from tokenizer import mlm_generator
from tokenizer import tokenization
from tokenizer import repeated_ngram_mining

from tokenizer import snippets
from tokenizer.tokenizer import Tokenizer
import numpy as np
import tensorflow as tf
import math

# import jieba_fast as jieba
import re, json
import numpy as np
from tokenizer.utils import text_token_mapping
from repeated_ngram_mask.tpu_dataset import StreamingFilesDataset
from tensorflow.python.data.ops import dataset_ops

import collections
_DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tf.disable_v2_behavior()

class PretrainGenerator(data_generator.DataGenerator):
  def __init__(self, 
      vocab_path,
      batch_size=32, 
      buffer_size=128, 
      do_lower_case=True,
      max_length=512,
      doc_stride=64,
      mask_ratio=0.15,
      random_ratio=0.1,
      min_tok=3,
      max_tok=10,
      mask_id=103,
      cls_id=101,
      sep_id=102,
      pad_id=0,
      geometric_p=0.1,
      max_pair_targets=10,
      random_next_sentence=False,
      max_predictions_per_seq=78,
      break_mode='sentence',
      doc_num=5
      ):

    self.tokenizer = Tokenizer(vocab_path, do_lower_case=do_lower_case)    
    self.max_length = max_length
    self.doc_stride = doc_stride
    self.batch_size = batch_size
    self.buffer_size = buffer_size
    self.max_length = max_length
    self.mask_ratio = mask_ratio
    self.random_ratio = random_ratio
    self.min_tok = min_tok
    self.max_tok = max_tok
    self.mask_id = mask_id
    self.pad_id = pad_id
    self.cls_id = cls_id
    self.sep_id = sep_id
    self.geometric_p = geometric_p
    self.max_pair_targets = max_pair_targets
    self.random_next_sentence = random_next_sentence
    self.break_mode = break_mode
    self.doc_num = doc_num
    self.max_predictions_per_seq = max_predictions_per_seq

    tf.logging.info("** random_next_sentence **")
    tf.logging.info(self.random_next_sentence)

    with tf.gfile.GFile(vocab_path, 'r') as frobj:
      self.vocab = []
      for index, line in enumerate(frobj):
        self.vocab.append(line.strip())

    self.mask_num = int(self.max_length * self.mask_ratio)

    self.mlm_gen = mlm_generator.MLMGenerator(
              mask_ratio=self.mask_ratio, 
              random_ratio=self.random_ratio,
              min_tok=self.min_tok,
              max_tok=self.max_tok,
              mask_id=self.mask_id,
              pad=self.pad_id,
              geometric_p=self.geometric_p,
              vocab=self.vocab,
              max_pair_targets=self.max_pair_targets)

    self.block_gen = sop_generator.BlockPairDataset(
          pad_id=self.pad_id,
          cls_id=self.cls_id,
          mask_id=self.mask_id,
          sep_id=self.sep_id,
    )

    tf.logging.info("** succeeded in initializing MLMGenerator and BlockPairDataset **")

  def iteration(self, data_path_dict, data_key):
    doc_lst = []
    count = 0
    with tf.gfile.GFile(data_path_dict['data_path'][data_key]['data'], "r") as frobj:
#     frobj = tf.gfile.GFile(data_path_dict['data_path'][data_key]['data'], "r")
      for line in frobj:
        if count == self.doc_num:
          for data_dict in self.preprocess(doc_lst):
            if data_dict:
              data_dict = self.postprocess(data_dict, use_tpu)
              yield data_dict
          doc_lst = []
          count = 0
        try:
          content = json.loads(line.strip())
          doc_lst.append(content)
          count += 1
        except:
          if doc_lst:
            for data_dict in self.preprocess(doc_lst):
              if data_dict:
                data_dict = self.postprocess(data_dict)
                yield data_dict
            doc_lst = []
            count = 0

  def mask_generation(self, sent_a_token_ids,
                      sent_b_token_ids, 
                      sent_rel_label,
                      entity_list_a,
                      entity_list_b,
                      ):
    span_lens_a = 0
    for item in entity_list_a:
      span_lens_a += (item[1] - item[0])

    span_lens_b = 0
    for item in entity_list_b:
      span_lens_b += (item[1] - item[0])

    nert_mask_prob_a = float(span_lens_a) / float(len(sent_a_token_ids))
    nert_mask_prob_b = float(span_lens_b) / float(len(sent_b_token_ids))

    [masked_sent_a, 
    masked_target_a, 
    pair_targets_a] = self.mlm_gen.ner_span_mask(
              sent_a_token_ids, 
              self.tokenizer,
              entity_spans=entity_list_a,
              return_only_spans=False,
              ner_masking_prob=1.0 if nert_mask_prob_a >= 0.15 else nert_mask_prob_a
             )

    [masked_sent_b, 
    masked_target_b, 
    pair_targets_b] = self.mlm_gen.ner_span_mask(
              sent_b_token_ids, 
              self.tokenizer,
              entity_spans=entity_list_b,
              return_only_spans=False,
              ner_masking_prob=1.0 if nert_mask_prob_b >= 0.15 else nert_mask_prob_b
             )

    masked_sent_a = masked_sent_a.tolist()
    masked_sent_b = masked_sent_b.tolist()

    origin_input = [self.cls_id] + sent_a_token_ids + [self.sep_id] + sent_b_token_ids + [self.sep_id]
    masked_input = [self.cls_id] + masked_sent_a + [self.sep_id] + masked_sent_b + [self.sep_id]
    input_mask = [1] * len(origin_input)
    segment_ids = [0] + [0]*len(masked_sent_a) + [0] + [1]*len(masked_sent_b) + [1]

    # sentence-a:add cls before sentence-a
    masked_lm_positions_a = [i+1 for i, e in enumerate(masked_target_a) if e != 0]
    # sentence-b:add cls+sent-a+sep before sentence-b
    masked_lm_positions_b = [i+2+len(masked_sent_a) for i, e in enumerate(masked_target_b) if e != 0]

    masked_lm_ids_a = [e for i, e in enumerate(masked_target_a) if e != 0]
    masked_lm_ids_b = [e for i, e in enumerate(masked_target_b) if e != 0]

    masked_lm_positions = masked_lm_positions_a + masked_lm_positions_b
    masked_lm_weights = [1]*len(masked_lm_positions)

    masked_lm_ids = masked_lm_ids_a + masked_lm_ids_b

    return {
      'origin_input': origin_input,
      'masked_input': masked_input,
      'input_mask': input_mask,
      'segment_ids': segment_ids,
      'masked_lm_positions': masked_lm_positions,
      'masked_lm_weights': masked_lm_weights,
      'masked_lm_ids': masked_lm_ids,
      'sent_rel_label_ids': sent_rel_label
    }
    
  def preprocess(self, doc_lst):
    
    token_lst = []
    size_lst = []
    for content in doc_lst:
      for sent in content:
        tokens = self.tokenizer.tokenize(sent)
        token_lst.extend(tokens)
        size_lst.append(len(tokens))
      size_lst.append(0)

    repeated_spans = repeated_ngram_mining.repeated_ngram_mining_v1(
                  token_lst, 
                  self.tokenizer, 
                  threshold=1)

    token_ids_lst = self.tokenizer.convert_tokens_to_ids(token_lst)
    doc_lst = self.block_gen.break_doc(
              token_ids_lst, 
              size_lst, 
              random_next_sentence=self.random_next_sentence,
              block_size=(self.max_length),
              short_seq_prob=0.1,
              break_mode=self.break_mode)

    for doc_index in doc_lst:
      sent_a_index = doc_index[0]
      sent_b_index = doc_index[1]

      if abs(sent_a_index[0]-sent_b_index[0]) <= 10 or abs(sent_a_index[1]-sent_b_index[1]) <= 10:
        continue

      sent_a_token_ids = token_ids_lst[sent_a_index[0]:(sent_a_index[1]+1)]
      sent_b_token_ids = token_ids_lst[sent_b_index[0]:(sent_b_index[1]+1)]

      sent_rel_label = doc_index[-1]

      span_dict_a = {}
      span_dict_b = {}
      for span in repeated_spans:
        if span[1] <= sent_a_index[1] and span[0] >= sent_a_index[0]:
          actual_span = (span[0]-sent_a_index[0], span[1]-sent_a_index[0])
          if actual_span not in span_dict_a:
            span_dict_a[actual_span] = 0
        if span[1] <= sent_b_index[1] and span[0] >= sent_b_index[0]:
          actual_span = (span[0]-sent_b_index[0], span[1]-sent_b_index[0])
          if actual_span not in span_dict_b:
            span_dict_b[actual_span] = 0

      mlm_dict = self.mask_generation(sent_a_token_ids,
                      sent_b_token_ids, 
                      sent_rel_label,
                      list(span_dict_a.keys()),
                      list(span_dict_b.keys()))

      yield mlm_dict

  def postprocess(self, tmp_dict):
    keys = list(tmp_dict.keys())
    for key in ['origin_input', 
                'masked_input', 
                'input_mask',
                'segment_ids',
                ]:
      tmp_dict[key] += [0]*(self.max_length-len(tmp_dict[key]))

    for key in ['masked_lm_positions', 
                'masked_lm_weights', 
                'masked_lm_ids'
                ]:
      tmp_dict[key] += [0]*(self.max_predictions_per_seq-len(tmp_dict[key]))
    
    tmp_list = []
    for key in [
                'origin_input', 
                'masked_input', 
                'input_mask',
                'segment_ids',
                'masked_lm_positions', 
                'masked_lm_weights', 
                'masked_lm_ids',
                'sent_rel_label_ids'
              ]:
    
      tmp_list.append(tmp_dict[key])
    return tuple(tmp_list)

  def _map_to_dict(self, 
                origin_input, 
                masked_input,
                input_mask,
                segment_ids,
                masked_lm_positions,
                masked_lm_weights,
                masked_lm_ids,
                sent_rel_label_ids):
    record_dict = {}
    record_dict['origin_input'] = origin_input
    record_dict['masked_input'] = masked_input
    record_dict['input_mask'] = input_mask
    record_dict['segment_ids'] = segment_ids
    record_dict['masked_lm_positions'] = masked_lm_positions
    record_dict['masked_lm_weights'] = masked_lm_weights
    record_dict['masked_lm_ids'] = masked_lm_ids
    record_dict['sent_rel_label_ids'] = sent_rel_label_ids
    return record_dict

  def to_dataset_(self, data_path_dict, data_key, types, shapes, names=None, padded_batch=False,
              is_training=False):
    def generator():
      for d in self.iteration(data_path_dict, data_key):
        yield d

    def gen_dataset(dummy):
      dataset = tf.data.Dataset.from_generator(
                generator, output_types=types, output_shapes=shapes)
      if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self.buffer_size)
      try:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      except:
        dataset = dataset.prefetch(self.batch_size*100)
      return dataset

    source_dataset = dataset_ops.Dataset.range(10)
    dataset = StreamingFilesDataset(
        source_dataset, filetype=gen_dataset)

    return dataset

  def to_dataset(self, data_path_dict, types, shapes, names=None, padded_batch=False,
              data_prior_dict={}, is_training=False,
              dataset_merge_method='sample',
              distributed_mode=None,
              worker_count=None,
              task_index=0):

    dataset_list = []
    data_prior = []
    for data_key in data_path_dict['data_path']:
      dataset = self.to_dataset_(data_path_dict, data_key, 
                    types, shapes, names=names, 
                    padded_batch=padded_batch,
                    is_training=is_training)
      dataset_list.append(dataset)
      if data_prior_dict:
        data_prior.append(data_prior_dict[data_key])

    if dataset_merge_method == 'sample':
      if not data_prior:
        p = tf.cast(np.array([1.0/len(dataset_list)]*len(dataset_list)), tf.float32)
      else:
        data_prior_np = np.array(data_prior)
        if np.sum(data_prior_np) != 1:
          data_prior_np = data_prior_np / np.sum(data_prior_np)
        tf.logging.info("** data prior **")
        tf.logging.info(data_prior_np)
        p = tf.cast(np.array(data_prior_np), tf.float32)

      combined_dataset = tf.contrib.data.sample_from_datasets(dataset_list, p)
      combined_dataset = combined_dataset.repeat()
      tf.logging.info("** sample_from_datasets **")
    elif dataset_merge_method == 'concat':
      combined_dataset = dataset_list[0]
      for i in range(1, len(dataset_list)):
        combined_dataset = combined_dataset.concatenate(dataset_list[i])
    tf.logging.info("** batch_size **")
    tf.logging.info(self.batch_size*len(dataset_list))
    combined_dataset = combined_dataset.batch(self.batch_size*len(dataset_list), drop_remainder=True)
    if distributed_mode == 'collective_reduce':
      if worker_count and task_index:
        combined_dataset = combined_dataset.shard(worker_count, task_index)
        tf.logging.info("** shard dataset for collective reduce **")
    tf.logging.info("** succeeded in building multiple-dataset **")
    
    combined_dataset = combined_dataset.map(
              lambda a0,a1,a2,a3,a4,a5,a6,a7:
              self._map_to_dict(a0, a1, a2, a3, a4, a5, a6, a7))
    return combined_dataset