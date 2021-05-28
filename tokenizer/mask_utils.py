
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