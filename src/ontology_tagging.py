from os import listdir
from os.path import isfile, join
import re
import nltk
import sys, http.client, urllib.request, urllib.parse, urllib.error, json
from nltk.corpus import treebank
from nltk.corpus import wordnet as wn
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
import gensim.downloader as api
from pprint import pprint
from file_locations import TRAINING_PATH, TEST_PATH, COMMON_WORDS_PATH

# constants
TRAIN_SENTS = treebank.tagged_sents()
UNIGRAM = UnigramTagger(TRAIN_SENTS, backoff=DefaultTagger('NN'))
BIGRAM = BigramTagger(TRAIN_SENTS, backoff=UNIGRAM)
TRIGRAM = TrigramTagger(TRAIN_SENTS, backoff=BIGRAM)
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
IRRELEVANT_WORDS = ["talk", "seminar", "lecture"]
WORD_VECTORS = api.load("glove-wiki-gigaword-100")
# the files to be trained on
TRAINING_FILES = [f for f in listdir(TRAINING_PATH) if isfile(join(TRAINING_PATH, f))]
# load the files to be tested
TEST_FILES = [f for f in listdir(TEST_PATH) if isfile(join(TEST_PATH, f))]

# code to convert POS tags into the right form for lemmatization
# https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
POS_TO_WORDNET = {

    'JJ': wn.ADJ,
    'JJR': wn.ADJ,
    'JJS': wn.ADJ,
    'RB': wn.ADV,
    'RBR': wn.ADV,
    'RBS': wn.ADV,
    'NN': wn.NOUN,
    'NNP': wn.NOUN,
    'NNS': wn.NOUN,
    'NNPS': wn.NOUN,
    'VB': wn.VERB,
    'VBG': wn.VERB,
    'VBD': wn.VERB,
    'VBN': wn.VERB,
    'VBP': wn.VERB,
    'VBZ': wn.VERB,

}

# manually created ontology tree
TREE = {"science": {},
        "maths": {},
        "engineering": {},
        "medicine": {},
        }


# reads in the list of the 1000 most common English words
def read_common_words():
    file = open(COMMON_WORDS_PATH, "r")
    file_text = file.read()
    words = []
    for line in file_text:
        words.append(line)

    return words


# pulls all the relevant information from the email
def pull_info(text, common_words):
    words = []
    tokenised = nltk.word_tokenize(text.lower())
    tokenised = [token for token in tokenised if token not in STOPWORDS]
    for token in tokenised:
        words.append(token)
    relevant_words = [i for i in tokenised + common_words if i not in common_words]
    tagged = TRIGRAM.tag(relevant_words)
    return tagged


# check to see if any of the words in the tree are in the email
def check_tree(text, current_tree):
    words = []
    for key in current_tree:
        if (key.lower() in text.lower()):
            words.append(key)
        words += check_tree(text, current_tree[key])
    return words


# get the lemmas of all the words pulled from the email
def get_lemmas(words):
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    lemmas = []
    for word in words:
        part_of_speech = ""
        try:
            part_of_speech = POS_TO_WORDNET[word[1]]
        except:
            pass
        if (part_of_speech != ""):
            lemma = lemmatizer.lemmatize(word[0], pos=part_of_speech)
            if (lemma.lower() not in IRRELEVANT_WORDS and re.search(r'\d', lemma) is None):
                lemmas.append(lemma.lower())
    return lemmas


# compares the similarity between two words using FastText
def get_similarity(email_word, tree_word):
    try:
        sim_score = WORD_VECTORS.similarity(email_word, tree_word)
    except:
        sim_score = 0
    return sim_score


# finds hyponyms and extends the ontology tree
def extend_tree(current_tree, count_depth):
    for key in current_tree:
        if (len(current_tree[key]) == 0 and count_depth > 0):
            synsets = wn.synsets(key)
            new_branches = {}
            for synset in synsets:
                hyponyms = synset.hyponyms()
                for hyponym in hyponyms:
                    lemmas = hyponym.lemmas()
                    for lemma in lemmas:
                        multiple_word = lemma.name().split('_')
                        if (len(multiple_word) == 1):
                            new_branches[lemma.name()] = new_branches[lemma.name()] = {}
                        else:
                            first_element = multiple_word[0]
                            multiple_word = multiple_word[1:]
                            furthest_branch = {}
                            # last_branch = {}
                            multiple_word.reverse()
                            for word in multiple_word:
                                current_branch = {}
                                current_branch[word] = furthest_branch
                                furthest_branch = current_branch
                            new_branches[first_element] = current_branch
            current_tree[key] = new_branches
            current_tree[key] = extend_tree(current_tree[key], count_depth - 1)
    return current_tree


# extend the tree
new_tree = extend_tree(TREE, 2)
pprint(new_tree)

# loops though each email and appropriately classifies it
for file in TEST_FILES:
    test_file = str(file)
    file_text = open(TEST_PATH + "/" + test_file, "r").read()
    common_words = read_common_words()
    words = pull_info(file_text, common_words)
    lemmas = get_lemmas(words)
    lemmas += check_tree(file_text, TREE)
    current_tree = new_tree
    classified = False
    tree_acc = []
    saved_sim_score = 0
    # loop until the best tag is found
    while (classified == False):
        # get the highest average similarity score of each node in the next level of the tree
        best_key = ""
        highest_sim_score = 0
        sim_score = 0
        for key in current_tree:
            count_lemmas = 0
            sim_avg = 0
            for lemma in lemmas:
                sim_avg += get_similarity(lemma, key)
                count_lemmas += 1
            try:
                sim_avg = sim_avg / count_lemmas
            except:
                sim_avg = 0
            if (sim_avg > highest_sim_score):
                best_key = key
                highest_sim_score = sim_avg
        # if the current tag has a higher score then we don't proceed
        if (highest_sim_score > saved_sim_score):
            if (best_key != ""):
                tree_acc.append(best_key)
                saved_sim_score = highest_sim_score
                current_tree = current_tree[best_key]
                if (len(current_tree) == 0):
                    classified = True
            else:
                classified = True
        else:
            classified = True
    print(test_file)
    print(str(tree_acc))
