from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag.perceptron import PerceptronTagger
import string
import re
from langid.langid import LanguageIdentifier, model

# For our Spark UDFs, all functions must accept and return strings (or basic types).\
# Adapted and optimized, based on Lab 2
tagger = PerceptronTagger()
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
stops = set(stopwords.words("english"))
lmtzr = WordNetLemmatizer()

def check_lang(data_str):
    #Detecting language using langid and return 'en' or 'NA'. 
    predict_lang = identifier.classify(data_str) # Calling the language identifier on the input string.
    if predict_lang[1] >= .9: # Checks if the confidence is high enough (>= 0.9)
        language = predict_lang[0]
    else:
        language = 'NA'
    return language

def remove_stops(data_str):
    # Removing English stopwords from string.
    if not data_str or not isinstance(data_str, str): # prevents errors from trying to process invalid or missing input
        return ""
    text = data_str.split() # Splits the input string by whitespace into a list of words
    cleaned_str = " ".join([word for word in text if word not in stops]) # joins the filtered words back into a single string 
    return cleaned_str

def remove_features(data_str):
    # compile regex
    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    num_re = re.compile('(\\d+)')
    mention_re = re.compile('@(\w+)')
    alpha_num_re = re.compile("^[a-z0-9_.]+$")
    # convert to lowercase
    data_str = data_str.lower()
    # remove hyperlinks
    data_str = url_re.sub(' ', data_str)
    # remove @mentions
    data_str = mention_re.sub(' ', data_str)
    # remove puncuation
    data_str = punc_re.sub(' ', data_str)
    # remove numeric 'words'
    data_str = num_re.sub(' ', data_str)
    # remove non a-z 0-9 characters and words shorter than 3 characters
    list_pos = 0
    cleaned_str = ''
    for word in data_str.split():
        if list_pos == 0:
            if alpha_num_re.match(word) and len(word) > 2:
                cleaned_str = word
            else:
                cleaned_str = ' '
        else:
            if alpha_num_re.match(word) and len(word) > 2:
                cleaned_str = cleaned_str + ' ' + word
            else:
                cleaned_str += ' '
        list_pos += 1
    return cleaned_str

def tag_and_remove(data_str):
    #POS tag and keep only nouns, verbs, and adjectives.
    
    if not data_str or not isinstance(data_str, str): # ensures the function only processes valid input strings
        return ""
    
    nn_tags = ['NN', 'NNP', 'NNPS', 'NNS'] #noun tags
    jj_tags = ['JJ', 'JJR', 'JJS'] #  adjectives 
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # verbs
    
    nltk_tags = nn_tags + jj_tags + vb_tags
    
    # breaking string into 'words'
    text = data_str.split()
    
    tagged_text = tagger.tag(text)
    
    cleaned_str = " ".join([word for word, tag in tagged_text if tag in nltk_tags])
    return cleaned_str

def lemmatize(data_str):
    
    #Lemmatizing words to their root form with POS.
    if not data_str or not isinstance(data_str, str): # ensures the function only processes valid input strings
        return ""
    
    text = data_str.split()
    tagged_words = tagger.tag(text) # Applies part-of-speech tagging using tagger
    lemmas = []
    for word, tag in tagged_words:
        if tag and 'v' in tag.lower():
            lemma = lmtzr.lemmatize(word, pos='v')
        else:
            lemma = lmtzr.lemmatize(word, pos='n')
        lemmas.append(lemma)
    cleaned_str = " ".join(lemmas)
    return cleaned_str