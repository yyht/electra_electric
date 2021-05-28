from tokenizer.extract_chinese_and_punct import ChineseAndPunctuationExtractor
import re
extractor = ChineseAndPunctuationExtractor()

def is_whitespace(c):
  if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    return True
  return False

def flat_list(h_list):
  e_list = []

  for item in h_list:
    if isinstance(item, list):
      e_list.extend(flat_list(item))
    else:
      e_list.append(item)
  return e_list

def text_token_mapping(tokenizer, text_raw, is_chinese_mode=False):

  if is_chinese_mode:
    sub_text = []
    buff = ""
    flag_en = False
    flag_digit = False
    for char in text_raw:
      if extractor.is_chinese_or_punct(char):
        if buff != "":
          sub_text.append(buff)
          buff = ""
        sub_text.append(char)
        flag_en = False
        flag_digit = False
      else:
        if re.compile('\d').match(char):
          if buff != "" and flag_en:
            sub_text.append(buff)
            buff = ""
            flag_en = False
          flag_digit = True
          buff += char
        else:
          if buff != "" and flag_digit:
            sub_text.append(buff)
            buff = ""
            flag_digit = False
          flag_en = True
          buff += char

    if buff != "":
      sub_text.append(buff)
  else:
    k = 0
    temp_word = ""
    sub_text = []
    raw_doc_tokens = tokenizer.customize_tokenize(text_raw)
    for c in text_raw:
      if is_whitespace(c):
        continue
      else:
        temp_word += c
      if temp_word == raw_doc_tokens[k]:
        sub_text.append(temp_word)
        temp_word = ""
        k += 1

  tok_to_orig_index = []
  orig_to_tok_index = []
  tok_to_orig_start_index = []
  tok_to_orig_end_index = []
  all_doc_tokens = []
  text_tmp = ''
  for (i, token) in enumerate(sub_text):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenizer.tokenize(token)
    text_tmp += token
    for sub_token in sub_tokens:
      tok_to_orig_index.append(i)
      all_doc_tokens.append(sub_token)
      tok_to_orig_start_index.append(len(text_tmp) - len(token))
      tok_to_orig_end_index.append(len(text_tmp) - 1)

  return [all_doc_tokens,
          tok_to_orig_index, 
          orig_to_tok_index,
          tok_to_orig_start_index,
          tok_to_orig_end_index,
          text_tmp]

def _check_is_max_context(doc_spans, cur_span_index, position):
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)

def text_tokenize(tokenizer, text_raw):
  sub_text = []
  buff = ""
  flag_en = False
  flag_digit = False
  for char in text_raw:
    if extractor.is_chinese_or_punct(char):
      if buff != "":
        sub_text.append(buff)
        buff = ""
      sub_text.append(char)
      flag_en = False
      flag_digit = False
    else:
      if re.compile('\d').match(char):
        if buff != "" and flag_en:
          sub_text.append(buff)
          buff = ""
          flag_en = False
        flag_digit = True
        buff += char
      else:
        if buff != "" and flag_digit:
          sub_text.append(buff)
          buff = ""
          flag_digit = False
        flag_en = True
        buff += char

  if buff != "":
    sub_text.append(buff)
  all_doc_tokens = []
  for (i, token) in enumerate(sub_text):
    sub_tokens = tokenizer.tokenize(token)
    for sub_token in sub_tokens:
      all_doc_tokens.append(sub_token)
      
  return all_doc_tokens