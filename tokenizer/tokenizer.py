
try:
  from tokenizers import (ByteLevelBPETokenizer,
                CharBPETokenizer,
                SentencePieceBPETokenizer,
                BertWordPieceTokenizer)
except:
  from tokenizer import tokenization
  BertWordPieceTokenizer = None

from tokenizer.snippets import is_string, is_py2
import unicodedata, re

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
    self._do_lower_case = do_lower_case

    tf.logging.info("** succeeded in initializing tokenizers **")

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

  def customize_tokenize(self, input_text):
    temp_x = ""
    for c in input_text:
      if self._is_cjk_character(c) or self._is_punctuation(c) or self._is_space(c) or self._is_control(c):
        temp_x += " " + c + " "
      else:
        temp_x += c
    return temp_x.split()

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

  def rematch(self, text, tokens):
    if is_py2:
      text = unicode(text)

    if self._do_lower_case:
      text = text.lower()

    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
      if self._do_lower_case:
        ch = unicodedata.normalize('NFD', ch)
        ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
      ch = ''.join([
          c for c in ch
          if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
      ])
      normalized_text += ch
      char_mapping.extend([i] * len(ch))

    text, token_mapping, offset = normalized_text, [], 0
    for token in tokens:
      if self._is_special(token):
        token_mapping.append([])
      else:
        token = self.stem(token)
        start = text[offset:].index(token) + offset
        end = start + len(token)
        token_mapping.append(char_mapping[start:end])
        offset = end

    return token_mapping