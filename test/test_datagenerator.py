import os, sys, six, re, json
import logging
import numpy as np
from collections import defaultdict

is_py2 = six.PY2
if not is_py2:
  basestring = str

def is_string(s):
  """判断是否是字符串
  """
  return isinstance(s, basestring)


def strQ2B(ustring):
  """
  """
  rstring = ''
  for uchar in ustring:
    inside_code = ord(uchar)
    # 
    if inside_code == 12288:
      inside_code = 32
    # 
    elif (inside_code >= 65281 and inside_code <= 65374):
      inside_code -= 65248
    rstring += unichr(inside_code)
  return rstring


def string_matching(s, keywords):
  """
  """
  for k in keywords:
    if re.search(k, s):
      return True
  return False

def text_segmentate(text, maxlen, seps='\n', strips=None):
  """
  """
  text = text.strip().strip(strips)
  if seps and len(text) > maxlen:
    pieces = text.split(seps[0])
    text, texts = '', []
    for i, p in enumerate(pieces):
      if text and p and len(text) + len(p) > maxlen - 1:
        texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        text = ''
      if i + 1 == len(pieces):
        text = text + p
      else:
        text = text + p + seps[0]
      if text:
        texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
    return texts
  else:
    return [text]


try:
  from tokenizers import (ByteLevelBPETokenizer,
                CharBPETokenizer,
                SentencePieceBPETokenizer,
                BertWordPieceTokenizer)

except:
  from tokenizer import tokenization
  BertWordPieceTokenizer = None


class Tokenizer(object):
  def __init__(self, vocab_path, do_lower_case=True):
    if BertWordPieceTokenizer:
      self.tokenizer = BertWordPieceTokenizer(vocab_path,
                                       lowercase=do_lower_case,
                                          )
    else:
      self.tokenizer = tokenization.FullTokenizer(
          vocab_path, do_lower_case=do_lower_case
        )

  def tokenize(self, input_text):
    if BertWordPieceTokenizer:
      return self.tokenizer.encode(input_text, add_special_tokens=False).tokens
    else:
      return self.tokenizer.tokenize(input_text)

  def encode(self, input_text, add_special_tokens=False):
    input_tokens = self.tokenize(input_text)
    if add_special_tokens:
      input_tokens = ['[CLS]'] + input_tokens + ['[SEP]']
    input_token_ids = self.convert_tokens_to_ids(input_tokens)
    return input_token_ids

  def padded_to_ids(self, input_text, max_length):
    if len(input_text) > max_length:
      return input_text[:max_length]
    else:
      return input_text + [0]*(max_length-len(input_text))

  def convert_tokens_to_ids(self, input_tokens):
    if BertWordPieceTokenizer:
      token_ids = [self.tokenizer.token_to_id(token) for token in input_tokens]
    else:
      token_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
    return token_ids

  def convert_ids_to_tokens(self, input_ids):
    if BertWordPieceTokenizer:
      input_tokens = [self.tokenizer.id_to_token(ids) for ids in input_ids]
    else:
      input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
    return input_tokens

  def decode(self, input_tokens):
    text, flag = '', False
    for i, token in enumerate(input_tokens):
      if token[:2] == '##':
        text += token[2:]
      elif len(token) == 1 and self._is_cjk_character(token):
        text += token
      elif len(token) == 1 and self._is_punctuation(token):
        text += token
        text += ' '
      elif i > 0 and self._is_cjk_character(text[-1]):
        text += token
      else:
        text += ' '
        text += token
    text = re.sub(' +', ' ', text)
    text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
    punctuation = self._cjk_punctuation() + '+-/={(<['
    punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
    punctuation_regex = '(%s) ' % punctuation_regex
    text = re.sub(punctuation_regex, '\\1', text)
    text = re.sub('(\d\.) (\d)', '\\1\\2', text)

    return text.strip()

  @staticmethod
  def stem(token):
    """
    """
    if token[:2] == '##':
      return token[2:]
    else:
      return token

  @staticmethod
  def _is_space(ch):
    """
    """
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
        unicodedata.category(ch) == 'Zs'

  @staticmethod
  def _is_punctuation(ch):
    """
    """
    code = ord(ch)
    return 33 <= code <= 47 or \
        58 <= code <= 64 or \
        91 <= code <= 96 or \
        123 <= code <= 126 or \
        unicodedata.category(ch).startswith('P')

  @staticmethod
  def _cjk_punctuation():
      return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

  @staticmethod
  def _is_cjk_character(ch):
    """
    reference：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    """
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
        0x3400 <= code <= 0x4DBF or \
        0x20000 <= code <= 0x2A6DF or \
        0x2A700 <= code <= 0x2B73F or \
        0x2B740 <= code <= 0x2B81F or \
        0x2B820 <= code <= 0x2CEAF or \
        0xF900 <= code <= 0xFAFF or \
        0x2F800 <= code <= 0x2FA1F

  @staticmethod
  def _is_control(ch):
    """
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')

  @staticmethod
  def _is_special(ch):
    """
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')


class DataGenerator(object):
  """template for DataGenerator
  """
  def __init__(self, batch_size=32, buffer_size=None):
    self.batch_size = batch_size
    self.buffer_size = buffer_size or batch_size * 1000

  def iteration(self, data_path_dict):
    raise NotImplementedError

  def to_dataset(self, data_path_dict, types, shapes, names=None, padded_batch=False,
              is_training=False):
    """
    """
    if names is None:
      def generator():
        for d in self.iteration(data_path_dict):
          yield d
    else:

      def warps(key, value):
        output_dict = {}
        for key_name, value_name in zip(key, value):
          output_dict[key_name] = value_name
        return output_dict

      def generator():
        for d in self.iteration(data_path_dict):
          yield d

      types = warps(names, types)
      
      shapes = warps(names, shapes)

    if padded_batch:
      dataset = tf.data.Dataset.from_generator(
        generator, output_types=types
      )
      dataset = dataset.shuffle(self.buffer_size)
      dataset = dataset.padded_batch(self.batch_size, shapes)
    else:
      dataset = tf.data.Dataset.from_generator(
        generator, output_types=types, output_shapes=shapes
      )
      dataset = dataset.shuffle(self.buffer_size)
      dataset = dataset.batch(self.batch_size)

    if is_training:
      dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
#     dataset = dataset.apply(tf.data.experimental.ignore_errors())
    return dataset


import numpy as np

class ParagraphInfo(object):
  def __init__(self, vocab):
    self.vocab2id = {}
    self.id2vocab = {}
    for index, word in enumerate(vocab):
      self.vocab2id[word] = index
      self.id2vocab[index] = word

  def is_start_word(self, idx):
    if isinstance(idx, int):
      return not self.id2vocab[idx].startswith('##')
    elif isinstance(idx, str):
      idx = self.vocab2id[idx]
      return not self.id2vocab[idx].startswith('##')

  # def get_word_piece_map(self, sentence):
  #   return [self.is_start_word(i) for i in sentence]

  def get_word_piece_map(self, sentence):
    """
    sentence: word id of sentence,
    [[0,1], [7,10]]
    """
    word_piece_map = []
    for segment in sentence:
      if isinstance(segment, list):
        for index, idx in enumerate(segment):
          if index == 0 or is_start_word(idx):
            word_piece_map.append(True)
          else:
            word_piece_map.append(self.is_start_word(idx))
      else:
        word_piece_map.append(self.is_start_word(segment))
    return word_piece_map

  def get_word_at_k(self, sentence, left, right, k, word_piece_map=None):
    num_words = 0
    while num_words < k and right < len(sentence):
      # complete current word
      left = right
      right = self.get_word_end(sentence, right, word_piece_map)
      num_words += 1
    return left, right

  def get_word_start(self, sentence, anchor, word_piece_map=None):
    word_piece_map = word_piece_map if word_piece_map is not None else self.get_word_piece_map(sentence)
    left  = anchor
    while left > 0 and word_piece_map[left] == False:
        left -= 1
    return left
  # word end is next word start
  def get_word_end(self, sentence, anchor, word_piece_map=None):
    word_piece_map = word_piece_map if word_piece_map is not None else self.get_word_piece_map(sentence)
    right = anchor + 1
    while right < len(sentence) and word_piece_map[right] == False:
        right += 1
    return right

def pad_to_max(pair_targets, pad):
  max_pair_target_len = max([len(pair_tgt) for pair_tgt in pair_targets])
  for pair_tgt in pair_targets:
    this_len = len(pair_tgt)
    for i in range(this_len, max_pair_target_len):
      pair_tgt.append(pad)
  return pair_targets

def pad_to_len(pair_targets, pad, max_pair_target_len):
  for i in range(len(pair_targets)):
    pair_targets[i] = pair_targets[i][:max_pair_target_len]
    this_len = len(pair_targets[i])
    for j in range(max_pair_target_len - this_len):
      pair_targets[i].append(pad)
  return pair_targets

def merge_intervals(intervals):
  intervals = sorted(intervals, key=lambda x : x[0])
  merged = []
  for interval in intervals:
    # if the list of merged intervals is empty or if the current
    # interval does not overlap with the previous, simply append it.
    if not merged or merged[-1][1] + 1 < interval[0]:
      merged.append(interval)
    else:
    # otherwise, there is overlap, so we merge the current and previous
    # intervals.
      merged[-1][1] = max(merged[-1][1], interval[1])

  return merged

def bert_masking(sentence, mask, tokens, pad, mask_id):
  sentence = np.copy(sentence)
  sent_length = len(sentence)
  target = np.copy(sentence)
  mask = set(mask)
  for i in range(sent_length):
    if i in mask:
      rand = np.random.random()
      if rand < 0.8:
        sentence[i] = mask_id
      elif rand < 0.9:
        # sample random token according to input distribution
        sentence[i] = np.random.choice(tokens)
    else:
      target[i] = pad
  return sentence, target, None

def span_masking(sentence, spans, tokens, pad, mask_id, pad_len, mask, replacement='word_piece', endpoints='external'):
  """
  pair_targets：
  [0:1]: masked start and end pos
  [1:]: masked word
  """
  sentence = np.copy(sentence)
  sent_length = len(sentence)
  target = np.full(sent_length, pad)
  pair_targets = []
  spans = merge_intervals(spans)
  assert len(mask) == sum([e - s + 1 for s,e in spans])
  # print(list(enumerate(sentence)))
  for start, end in spans:
    lower_limit = 0 if endpoints == 'external' else -1
    upper_limit = sent_length - 1 if endpoints == 'external' else sent_length
    if start > lower_limit and end < upper_limit:
      if endpoints == 'external':
        pair_targets += [[start - 1, end + 1]]
      else:
        pair_targets += [[start, end]]
      pair_targets[-1] += [sentence[i] for i in range(start, end + 1)]
    rand = np.random.random()
    for i in range(start, end + 1):
      assert i in mask
      target[i] = sentence[i]
      if replacement == 'word_piece':
        rand = np.random.random()
      if rand < 0.8:
        sentence[i] = mask_id
      elif rand < 0.9:
        # sample random token according to input distribution
        sentence[i] = np.random.choice(tokens)
  pair_targets = pad_to_len(pair_targets, pad, pad_len + 2)
  # if pair_targets is None:
  return sentence, target, pair_targets

tokenizer = Tokenizer(vocab_path="/data/albert/asr/test/vocab_bert_cn.txt", do_lower_case=True)
vocab = []
with open("/data/albert/asr/test/vocab_bert_cn.txt") as frobj:
    for index, line in enumerate(frobj):
        vocab.append(line.strip())

s = ParagraphInfo(vocab)

import numpy as np
import tensorflow as tf
from collections import OrderedDict, Counter

import re
CH_PUNCTUATION = u"['\\\\,\\!#$%&\'()*+-/:￥;<=>.?\\。n@[\\]^▽_`{|}~'－＂＃＄％＆＇，：；＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡]"

def compresoneadjtuple(s):
  """useful to compress adjacent entries"""
  if len(s) < 1: return s, True
  finals=[]
  for pos in range(len(s)-1):
    firstt, secondt = s[pos],s[pos+1]
    # if (firstt[1]==secondt[0]) or (firstt[1]+1==secondt[0]):
    if (firstt[1] == secondt[0]):
      finals.append((firstt[0],secondt[1]))
      finals.extend(s[pos+2:])
      return finals, False
    else:
      finals.append(firstt)
  finals.append(s[-1])
  return finals, True

def merge_intervals(intervals):
  s = sorted(intervals, key=lambda t: t[0])
  m = 0
  for  t in s:
    if t[0] >= s[m][1]:
      m += 1
      s[m] = t
    else:
      s[m] = (s[m][0], max([t[1], s[m][1]]))
  done = False
  s = s[:m+1]
  while(not done):
    s, done = compresoneadjtuple(s)
  return s

def repeated_ngram_mining(input_text, tokenizer, 
                  threshold=2):
  """
  http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/AudioMotif_accept.pdf
  """
  if not isinstance(input_text, list):
    sentence = tokenizer.tokenize(input_text)
  else:
    sentence = input_text
  
  hash_table = OrderedDict({})
  for index, token in enumerate(sentence):
    a = re.search(CH_PUNCTUATION,  token)
    if a:
      continue
    if token in hash_table:
      hash_table[token].append(index)
    else:
      hash_table[token] = [index]

  delta_table = OrderedDict({})
  for token in hash_table:
    for i, _ in enumerate(hash_table[token]):
      for j, _ in enumerate(hash_table[token]):
        if j > i:
          tmp = hash_table[token][j] - hash_table[token][i]
          if tmp in delta_table:
            delta_table[tmp].append((hash_table[token][i], hash_table[token][j], token))
          else:
            delta_table[tmp] = [(hash_table[token][i], hash_table[token][j], token)]
                
  histogram = [[]]
 
  for key in delta_table:
    if len(delta_table[key]) >= 1 and key not in [0]:
      delta_table[key] = sorted(delta_table[key], key=lambda item:item[1])
      for prev_index  in range(len(delta_table[key])-1):
        cur_index = prev_index + 1
        if delta_table[key][cur_index][0]  - delta_table[key][prev_index][0] <= threshold:
          histogram[-1].append(delta_table[key][prev_index])
        else:
          if delta_table[key][prev_index] not in histogram[-1]:
            histogram[-1].append(delta_table[key][prev_index])
          histogram.append([])
      if histogram[-1]:
        if delta_table[key][cur_index] not in histogram[-1]:
          histogram[-1].append(delta_table[key][cur_index])
        histogram.append([])
  ngram = []
  for item in histogram:
    if not item:
      continue
    if item[-1][0] - item[0][0] >= 1:
      if (item[0][0], item[-1][0]) not in ngram:
        ngram.append((item[0][0], item[-1][0]))
    if item[-1][1] - item[0][1] >= 1:
      if (item[0][1], item[-1][1]) not in ngram:
        ngram.append((item[0][1], item[-1][1]))
                
  merged_ngram = merge_intervals(ngram)
  return merged_ngram


import numpy as np
import tensorflow as tf
import jieba_fast as jieba
import math
"""
https://github.com/facebookresearch/SpanBERT/blob/master/pretraining/fairseq/data/masking.py
"""

class MLMGenerator(object):
  def __init__(self, 
              mask_ratio, 
              random_ratio,
              min_tok,
              max_tok,
              mask_id,
              pad,
              geometric_p,
              vocab,
               max_pair_targets,
               replacement_method='word_piece',
              endpoints='',
              **kargs
              ):
    self.mask_ratio = mask_ratio
    self.random_ratio = random_ratio
    self.min_tok = min_tok
    self.max_tok = max_tok
    self.mask_id = mask_id
    self.pad = pad
    self.tokens = [index for index, word in enumerate(vocab)]
    self.geometric_p = geometric_p
    self.max_pair_targets = max_pair_targets

    self.p = geometric_p
    self.len_distrib = [self.p * (1-self.p)**(i - self.min_tok) for i in range(self.min_tok, self.max_tok + 1)] if self.p >= 0 else None
    self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]
    self.lens = list(range(self.min_tok, self.max_tok + 1))
    self.paragraph_info = ParagraphInfo(vocab)
    self.replacement_method = replacement_method
    """
    if endpoints is external: index_range = np.arange(a+1, b)
    else: index_range = np.arange(a, b+1)
    """
    self.endpoints = endpoints
    self.max_pair_targets = max_pair_targets

  def random_mask(self, input_text, 
                  tokenizer, **kargs):
    if not isinstance(input_text, list):
      sentence = tokenizer.encode(input_text, add_special_tokens=False)
    else:
      sentence = input_text
    sent_length = len(sentence)
    mask_num = math.ceil(sent_length * self.mask_ratio)
    mask = np.random.choice(sent_length, mask_num, replace=False)
    return bert_masking(sentence, mask, self.tokens, self.pad, self.mask_id)

  def mask_entity(self, sentence, mask_num, word_piece_map, spans, mask, entity_spans):
    if len(entity_spans) > 0:
      entity_span = entity_spans[np.random.choice(range(len(entity_spans)))]
      spans.append([entity_span[0], entity_span[0]])
      for idx in range(entity_span[0], entity_span[1] + 1):
        if len(mask) >= mask_num:
          break
        spans[-1][-1] = idx
        mask.add(idx)

  def mask_random_span(self, sentence, mask_num, word_piece_map, spans, mask, span_len, anchor):
    # find word start, end
    # this also apply ngram and whole-word-mask for english
    left1, right1 = self.paragraph_info.get_word_start(sentence, anchor, word_piece_map), self.paragraph_info.get_word_end(sentence, anchor, word_piece_map)
    spans.append([left1, left1])
    for i in range(left1, right1):
      if len(mask) >= mask_num:
        break
      mask.add(i)
      spans[-1][-1] = i
    num_words = 1
    right2 = right1
    while num_words < span_len and right2 < len(sentence) and len(mask) < mask_num:
      # complete current word
      left2 = right2
      right2 = self.paragraph_info.get_word_end(sentence, right2, word_piece_map)
      num_words += 1
      for i in range(left2, right2):
        if len(mask) >= mask_num:
          break
        mask.add(i)
        spans[-1][-1] = i

  def ner_span_mask(self, 
                input_text, 
                tokenizer,
                entity_spans=None,
                return_only_spans=False,
                ner_masking_prob=0.1,
                threshold=1,
                **kargs):
    """mask tokens for masked language model training
    Args:
        sentence: 1d tensor, token list to be masked
        mask_ratio: ratio of tokens to be masked in the sentence
    Return:
        masked_sent: masked sentence
    """
    if not isinstance(input_text, list):
      sentence = tokenizer.encode(input_text, add_special_tokens=False)
    else:
      sentence = input_text
    sent_length = len(sentence)
    mask_num = math.ceil(sent_length * self.mask_ratio)
    mask = set()
    word_piece_map = self.paragraph_info.get_word_piece_map(sentence)
    spans = []
    
    while len(mask) < mask_num:
      if entity_spans:
        if np.random.random() <= ner_masking_prob:
          self.mask_entity(sentence, mask_num, word_piece_map, spans, mask, entity_spans)
        else:
          span_len = np.random.choice(self.lens, p=self.len_distrib)
          anchor  = np.random.choice(sent_length)
          if anchor in mask:
            continue
          self.mask_random_span(sentence, mask_num, word_piece_map, spans, mask, span_len, anchor)
      else:
        span_len = np.random.choice(self.lens, p=self.len_distrib)
        anchor  = np.random.choice(sent_length)
        if anchor in mask:
          continue
        self.mask_random_span(sentence, mask_num, word_piece_map, spans, mask, span_len, anchor)
    sentence, target, pair_targets = span_masking(sentence, spans, self.tokens, self.pad, self.mask_id, self.max_pair_targets, mask, replacement=self.replacement_method, endpoints=self.endpoints)
    if return_only_spans:
      pair_targets = None
    return sentence, target, pair_targets

  def repeated_ngram_mask(self, input_text,
                        tokenizer,
                        return_only_spans=False,
                        ner_masking_prob=0.5,
                        threshold=1,
                        **kargs):
    if not isinstance(input_text, list):
      sentence = tokenizer.encode(input_text, add_special_tokens=False)
      entity_spans = []
    else:
      sentence = input_text

    repeated_spans = repeated_ngram_mining(
                  input_text, 
                  tokenizer, 
                  threshold=threshold)

    [sentence, 
    target, 
    pair_targets] = self.ner_span_mask(
                sentence,
                tokenizer, 
                entity_spans=repeated_spans,
                return_only_spans=return_only_spans,
                ner_masking_prob=ner_masking_prob,
                **kargs)

    return sentence, target, pair_targets








        


mlm = MLMGenerator(
                mask_ratio=0.15, 
              random_ratio=0.1,
              min_tok=3,
              max_tok=10,
              mask_id=103,
              pad=0,
              geometric_p=0.1,
              vocab=vocab,
max_pair_targets=10)

text = """
新华社北京4月29日电 4月29日11时23分，中国空间站天和核心舱发射升空，准确进入预定轨道，任务取得成功。中共中央总书记、国家主席、中央军委主席习近平致贺电，代表党中央、国务院和中央军委，向载人航天工程空间站阶段飞行任务总指挥部并参加天和核心舱发射任务的各参研参试单位和全体同志致以热烈的祝贺和诚挚的问候。
"""
mlm.random_mask(text, 
                  data_gen.tokenizer)

text = """

联播+历史是最好的老师。历史在人民的探索、奋斗中造就了中国共产党，党团结带领人民群众书写了中华民族发展新篇章。

习近平总书记十分重视在全党开展党史学习教育，将党史学习教育的主要内容概括为16个字，即“学史明理、学史增信、学史崇德、学史力行”。

何谓学史明理？习近平总书记指出，明理是增信、崇德、力行的前提。在党史学习教育中，要深刻领悟中国共产党为什么能、马克思主义为什么行、中国特色社会主义为什么好等道理。

通过党史学习教育，我们方能在乱云飞渡中找准方向，在风险挑战前砥砺胆识，开创属于当代人的历史伟业。央视网《联播+》特梳理总书记重要讲话，与您一起学习。
"""

entity_map = [(0, 1),
 (3, 4),
 (12, 13),
 (15, 16),
 (26, 31),
 (38, 39),
 (44, 45),
 (55, 60),
 (70, 75),
 (78, 83),
 (98, 101),
 (103, 106),
 (108, 111),
 (113, 116),
 (121, 124),
 (126, 131),
 (135, 136),
 (138, 139),
 (141, 142),
 (144, 145),
 (151, 156),
 (164, 171),
 (177, 181),
 (184, 185),
 (190, 194),
 (202, 207),
 (243, 244),
 (252, 253),
 (259, 261),
 (271, 272)]

f = mlm.ner_span_mask(text, 
              data_gen.tokenizer,
            entity_spans=entity_map,
                      return_only_spans=False,
                      ner_masking_prob=1
             )

mlm.repeated_ngram_mask(text, 
              data_gen.tokenizer,
                      return_only_spans=False,
                      ner_masking_prob=1
             )


import json
import numpy as np

"""
https://github.com/facebookresearch/SpanBERT/blob/master/pretraining/fairseq/data/span_bert_dataset.py
"""

class BlockPairDataset(object):
  def __init__(
    self,
    pad_id,
    cls_id,
    mask_id,
    sep_id):

    self.pad_id = pad_id
    self.cls_id = cls_id
    self.mask_id = mask_id
    self.sep_id = sep_id

  def sentence_mode(self, tokens, sizes, 
          random_next_sentence,
          block_size,
          short_seq_prob):
    assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
    curr = 0
    block_indices = []
    for sz in sizes:
      if sz == 0:
        continue
      block_indices.append((curr, curr + sz))
      curr += sz
    max_num_tokens = block_size - 3 # Account for [CLS], [SEP], [SEP]

    sent_pairs = []
    sizes = []
    target_seq_length = max_num_tokens
    if np.random.random() < short_seq_prob:
      target_seq_length = np.random.randint(2, max_num_tokens)
    current_chunk = []
    current_length = 0
    curr = 0
    while curr < len(block_indices):
      sent = block_indices[curr]
      current_chunk.append(sent)
      current_length = current_chunk[-1][1] - current_chunk[0][0]
      if curr == len(block_indices) - 1 or current_length >= target_seq_length:
        if current_chunk:
          a_end = 1
          if len(current_chunk) > 2:
            a_end = np.random.randint(1, len(current_chunk) - 1)
          sent_a = current_chunk[:a_end]
          sent_a = (sent_a[0][0], sent_a[-1][1])
          next_sent_label = (
            1 if np.random.rand() > 0.5 else 0
          )
          if len(current_chunk) == 1 or (random_next_sentence and next_sent_label):
            target_b_length = target_seq_length - (sent_a[1] - sent_a[0])
            random_start = np.random.randint(0, len(block_indices) - len(current_chunk))
            # avoid current chunks
            # we do this just because we don't have document level segumentation
            random_start = (
              random_start + len(current_chunk)
              if block_indices[random_start][1] > current_chunk[0][0]
              else random_start
            )
            sent_b = []
            for j in range(random_start, len(block_indices)):
              sent_b = (
                (sent_b[0], block_indices[j][1])
                if sent_b else block_indices[j]
              )
              if block_indices[j][0] == current_chunk[0][0]:
                break
              # length constraint
              if sent_b[1] - sent_b[0] >= target_b_length:
                break
            num_unused_segments = len(current_chunk) - a_end
            curr -= num_unused_segments
            next_sent_label = 1
          elif not random_next_sentence and next_sent_label:
            # sop
            next_sent_label = 1
            sent_b = current_chunk[a_end:]
            sent_b = (sent_b[0][0], sent_b[-1][1])
            sent_a, sent_b = sent_b, sent_a
          else:
            next_sent_label = 0
            sent_b = current_chunk[a_end:]
            sent_b = (sent_b[0][0], sent_b[-1][1])
          sent_a, sent_b = self._truncate_sentences(sent_a, sent_b, max_num_tokens)
          sent_pairs.append((sent_a, sent_b, next_sent_label))
          if sent_a[0] >= sent_a[1] or sent_b[0] >= sent_b[1]:
            print(sent_a, sent_b)
          sizes.append(3 + sent_a[1] - sent_a[0] + sent_b[1] - sent_b[0])
        current_chunk = []
      curr += 1
    return sent_pairs, sizes 

  def doc_mode(self, tokens, sizes,
          random_next_sentence,
          block_size,
          short_seq_prob):
    assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
    curr = 0
    cur_doc = []
    block_indices = []
    for sz in sizes:
      if sz == 0:
        if len(cur_doc) == 0: continue
        block_indices.append(cur_doc)
        cur_doc = []
      else:
        cur_doc.append((curr, curr + sz))
      curr += sz
    max_num_tokens = block_size - 3 # Account for [CLS], [SEP], [SEP]
    
    sent_pairs = []
    sizes = []
    for doc_id, doc in enumerate(block_indices):
      current_chunk = []
      current_length = 0
      curr = 0
      target_seq_length = max_num_tokens
      short = False
      if np.random.random() < short_seq_prob:
        short = True
        target_seq_length = np.random.randint(2, max_num_tokens)
      while curr < len(doc):
        sent = doc[curr]
        current_chunk.append(sent)
        current_length = current_chunk[-1][1] - current_chunk[0][0]
        if curr == len(doc) - 1 or current_length >= target_seq_length:
          if current_chunk:
            a_end = 1
            if len(current_chunk) > 2:
              a_end = np.random.randint(1, len(current_chunk) - 1)
            sent_a = current_chunk[:a_end]
            sent_a = (sent_a[0][0], sent_a[-1][1])
            next_sent_label = (
              1 if np.random.rand() > 0.5 else 0
            )

            if len(current_chunk) == 1 or (random_next_sentence and next_sent_label):
              next_sent_label = 1
              target_b_length = target_seq_length - (sent_a[1] - sent_a[0])
              for _ in range(10):
                rand_doc_id = np.random.randint(0, len(block_indices) - 1)
                if rand_doc_id != doc_id:
                  break
              random_doc = block_indices[rand_doc_id]
              random_start = np.random.randint(0, len(random_doc))
              sent_b = []
              for j in range(random_start, len(random_doc)):
                sent_b = (
                  (sent_b[0], random_doc[j][1])
                  if sent_b else random_doc[j]
                )
                if sent_b[1] - sent_b[0] >= target_b_length:
                  break
              num_unused_segments = len(current_chunk) - a_end
              curr -= num_unused_segments
            elif not random_next_sentence and next_sent_label:
              next_sent_label = 1
              sent_b = current_chunk[a_end:]
              sent_b = (sent_b[0][0], sent_b[-1][1])
              sent_a, sent_b = sent_b, sent_a
            else:
              next_sent_label = 0
              sent_b = current_chunk[a_end:]
              sent_b = (sent_b[0][0], sent_b[-1][1])
            sent_a, sent_b = self._truncate_sentences(sent_a, sent_b, max_num_tokens)
            sent_pairs.append((sent_a, sent_b, next_sent_label))
            if sent_a[0] >= sent_a[1] or sent_b[0] >= sent_b[1]:
              print(sent_a, sent_b)
            sizes.append(3 + sent_a[1] - sent_a[0] + sent_b[1] - sent_b[0])
            #print(3 + sent_a[1] - sent_a[0] + sent_b[1] - sent_b[0], curr == len(doc) - 1, short)
          current_chunk = []
        curr += 1
    return sent_pairs, sizes

  def random_mode(self, tokens, sizes,
          random_next_sentence,
          block_size,
          short_seq_prob):
    block_size = block_size - 3
    block_size //= 2  # each block should have half of the block size since we are constructing block pair
    length = math.ceil(len(tokens) / block_size)

    def block_at(i):
      start = i * block_size
      end = min(start + block_size, len(tokens))
      return (start, end)

    block_indices = [block_at(i) for i in range(length)]

    sizes = np.array(
      # 2 block lengths + 1 cls token + 2 sep tokens
      # note: this is not accurate and larger than pairs including last block
      [block_size * 2 + 3] * len(block_indices)
    )
    return block_indices, sizes

  def _truncate_sentences(self, sent_a, sent_b, max_num_tokens):
    while True:
      total_length = sent_a[1] - sent_a[0] + sent_b[1] - sent_b[0]
      if total_length <= max_num_tokens:
        return sent_a, sent_b

      if sent_a[1] - sent_a[0] > sent_b[1] - sent_b[0]:
        sent_a = (
          (sent_a[0]+1, sent_a[1])
          if np.random.rand() < 0.5
          else (sent_a[0], sent_a[1] - 1)
        )
      else:
        sent_b = (
          (sent_b[0]+1, sent_b[1])
          if np.random.rand() < 0.5
          else (sent_b[0], sent_b[1] - 1)
        )

  def _rand_block_index(self, i, block_indices):
    """select a random block index which is not given block or next
       block
    """
    idx = np.random.randint(len(block_indices) - 3)
    return idx if idx < i else idx + 2

  def break_doc(self, tokens, sizes, 
              random_next_sentence,
              block_size,
              short_seq_prob,
              break_mode):
    if break_mode == "sentence":
      [sent_pairs, sizes] = self.sentence_mode(
                            tokens, sizes,
                            random_next_sentence,
                            block_size,
                            short_seq_prob)
    elif break_mode == 'doc':
      [sent_pairs, sizes] = self.doc_mode(
                            tokens, sizes,
                            random_next_sentence,
                            block_size,
                            short_seq_prob)
    else:
      [block_indices, sizes] = self.random_mode(
                            tokens, sizes,
                            random_next_sentence,
                            block_size,
                            short_seq_prob)
      
      sent_pairs = []
      for index in range(len(block_indices)-1):
        next_sent_label = (
          1 if np.random.rand() > 0.5 else 0
        )
        block1 = block_indices[index]
        if (next_sent_label and index == len(block_indices) - 1) or (next_sent_label and random_next_sentence):
          next_sent_label = 1
          block2 = block_indices[self._rand_block_index(index, block_indices)]
        elif next_sent_label and not random_next_sentence:
          block2 = block_indices[index+1]
          block1, block2 = block2, block1
          next_sent_label = 1
        else:
          next_sent_label = 0
          block2 = block_indices[index+1]
        sent_pairs.append((block1, block2, next_sent_label))

    return sent_pairs