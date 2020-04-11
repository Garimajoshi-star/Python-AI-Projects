# Before starting with the implementation of NLP in Python or Anaconda environment, install basic libraries shown as follows:
pip install nltk
pip install pattern
pip install genism
pip install spaCy
pip install TextBlob

# NLP environment will be created only after downloading the NLTK Downloader. To do so, open the Python cmd or development environment and type the following code:
import nltk
nltk.download()

# A graphical window will appear containing a list of packages required to carry out the process of NLP. Click the download button present on the left side of the window to download these packages


Let’s perform the following steps to implement tokenisation of words and sentences using the NLTK module:

#  Import libraries and call methods

import nltk
nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize

#Input text 
data = "Delhi is the capital of India. It covers an area of 1,484 square kilometres."

# Program 3: Tokenising 
phrases = sent_tokenize(data)
words = word_tokenize(data)
 
print("\n\n", phrases)
print("\n\n", words)

# Now, perform the following steps to predict Parts of Speech (POS) using NLTK libraries:
# Import libraries
import nltk
from nltk.tokenize import PunktSentenceTokenizer

# Predict POS 
data = "Delhi is the capital of India. It covers an area of 1,484 square kilometres."
sentences = nltk.sent_tokenize(data)   
for sent in sentences:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))

# Now perform the following steps to implement lemmatisation using NLTK libraries, PorterStemmer and WordNetLemmatizer modules:
# Import libraries
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

# Perform lemmatisation
wordnet_lemmatizer = WordNetLemmatizer()

data= "Delhi is the capital of India. It covers an area of 1,484 square kilometres."
punctuations=" ; , . ! ? : "

sentence_words = nltk.word_tokenize(data)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

sentence_words
print("{0:40}{1:40}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:40}{1:40}".format(word,wordnet_lemmatizer.lemmatize(word, pos="v")))


# Now perform the following steps to identify stop words using NLTK and Corpus modules:
# Import libraries

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Identify stop words

data = "Delhi is the capital of India. It covers an area of 1,484 square kilometres."
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
print("\n\n", wordsFiltered, "\n\n")
print(len(stopWords))

# Display stop words
print("\n\n", stopWords)

# now, perform the following steps to create dependency parsing and chunking using the NLTK library:
# Import library
import nltk

# Create a parse tree 

data = "Delhi is the capital of India." 
tokens = nltk.word_tokenize(data)
print(tokens)
tag = nltk.pos_tag(tokens)
print(tag)
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp  =nltk.RegexpParser(grammar)
parse_tree= cp.parse(tag)
print(parse_tree)
parse_tree.draw()   

# now perform the following steps to implement the Name Entity Recognition (NER) process using NLTK libraries and spaCy module:

# Install and download library
pip install spacy
python -m spacy download en

# Import libraries

import nltk
import spacy
import en_core_web_sm
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags

# Perform NER 

spacy_nlp = spacy.load('en')
data = "Delhi is the capital of India. It covers an area of 1,484 square kilometres.”
document = spacy_nlp(data)
print('Input data: %s' % (data))
for token in document.ents:
    print('Type: %s, Value: %s' % (token.label_, token))

Here you have learned the systematic process involved in the processing of natural language. 
