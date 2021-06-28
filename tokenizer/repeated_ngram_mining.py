import numpy as np
import tensorflow as tf
tf.disable_v2_behavior()
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