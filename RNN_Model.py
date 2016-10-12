vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENECE_START"
sentence_end_token = "SENTENECE_END"


import csv
import nltk.tokenize as nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import *

#APPEND start_token and end_token
print("Reading CSV file........")
with open('/home/prateek/Downloads/RNN- Machine Learning/data/reddit-comments-2015-08.csv', 'rb') as f:
	reader = csv.reader(f, skipinitialspace = True)
	#reader.next()
	sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
	# itertools.chain(*iterables)
	# nltk.sent_tokenize tokenize the text into sentences
	# ntlk.word_tokenize tokenize the sentences into words (IT AUTOMATICALLY ASSUMES THAT THE TEXT IS SENT_TOKENIZED)
	sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))	

tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

#dont directly pass the text, it will return characters. pass the text first through the tokenizer
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found %d unique word tokens.") % len(word_freq.items())

vocab = word_freq.most_common(vocabulary_size-1)
# most_common returns the list of 8000 most common words used
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary_size %d" % vocabulary_size
print "The The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])