import json

def load_json_file(filepath):
  with open(filepath) as json_file:
    return json.load(json_file)['data']

def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens

def is_whitespace(c):
    """Check if the passed string is a whitespace or similar stuff"""
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

def list_to_dict(l):
    """Convert a list to a dictionary with progressive index"""
    return {index: el for (index, el) in enumerate(l)}  