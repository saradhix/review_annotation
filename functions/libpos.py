import nltk
from nltk.corpus import stopwords
from nltk.stem import *

def get_adjectives(sentence):
  sentence = sentence.lower()
  pos_result = nltk.pos_tag(nltk.word_tokenize(sentence))
  adjs = set()
  #print pos_result
  for t in pos_result:
    if str(t[0]) in stopwords.words("english"):
      continue
    if str(t[1]).startswith('JJ'):
      adjs.add(t[0])
    #if str(t[1]).startswith('RB'):
      #adjs.add(t[0])
  #adjs = set([stemmer.stem(adj) for adj in adjs])
  return adjs
