
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