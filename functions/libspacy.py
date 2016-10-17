from spacy.parts_of_speech import ADV
from spacy.en import English
import spacy


nlp = English()
def get_adjectives(sentence):
  adjs = set()
  sentence = nlp(sentence.decode('utf-8'))
  for token in sentence:
    if token.pos == spacy.parts_of_speech.ADJ:
      adjs.add(str(token))
  return adjs
'''
s = "A healthy king lives happily"
print get_adjectives(s)
s = "I am very rich and beautiful girl"
print get_adjectives(s)
'''
'''
sentence = nlp(u'A healthy man lives happily')
print sentence
for token in sentence:
  print token, token.pos, is_adverb(token)
'''
