import os, sys, six, re, json
import logging
import numpy as np
from collections import defaultdict

is_py2 = six.PY2
if not is_py2:
  basestring = str

def is_string(s):
  """
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