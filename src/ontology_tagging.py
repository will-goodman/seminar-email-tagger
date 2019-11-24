from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
from nltk.corpus import treebank
from nltk.corpus import wordnet as wn
from file_locations import TRAINING_PATH, TEST_PATH, COMMON_WORDS_PATH
from os.path import isfile, join
from os import listdir
from pprint import pprint
import gensim.downloader as api
import re
import nltk
import os

TRAINING_SENTS = treebank.tagged_sents()
UNIGRAM = UnigramTagger(TRAINING_SENTS, backoff=DefaultTagger('NN'))
BIGRAM = BigramTagger(TRAINING_SENTS, backoff=UNIGRAM)
TRIGRAM = TrigramTagger(TRAINING_SENTS, backoff=BIGRAM)
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
WORD_VECTORS = api.load("glove-wiki-gigaword-100")
# the files to be trained on
TRAINING_FILES = [f for f in listdir(TRAINING_PATH) if isfile(join(TRAINING_PATH, f))]
# load the files to be tested
TEST_FILES = [f for f in listdir(TEST_PATH) if isfile(join(TEST_PATH, f))]

# Manual list of words to be considered "irrelevant"
IRRELEVANT_WORDS = ["talk", "seminar", "lecture"]

# manually created ontology tree, which is later extended
TREE = {
    "science": {},
    "maths": {},
    "engineering": {},
    "medicine": {}
}

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


def read_common_words():
    """
    Reads in the file containing the most common words.
    :return: A list of the most common English words.
    """
    most_common_words = []
    with open(COMMON_WORDS_PATH, "r") as words_file:
        for line in words_file.read():
            most_common_words.append(line)

    return most_common_words


def retrieve_tags(text, common_words):
    """
    Pulls all the tagged information from the email.
    :param text: The email text.
    :param common_words: A list of the most common words.
    :return: The tagged information, with any of the common words removed.
    """
    tokens = nltk.word_tokenize(text.lower())
    tokens_stopwords_removed = [token for token in tokens if token not in STOPWORDS]

    relevant_words = [i for i in tokens_stopwords_removed if i not in common_words]
    tagged = TRIGRAM.tag(relevant_words)
    return tagged


def check_tree(text, current_tree):
    """
    Checks to see if any of the words in the tree are in the email.
    :param text: The email text.
    :param current_tree: The tree of words to check.
    :return: A list of any words which are in the tree and email.
    """
    words_acc = []
    lowered_email_text = text.lower()
    for key in current_tree:
        if key.lower() in lowered_email_text:
            words_acc.append(key)
        words_acc.extend(check_tree(text, current_tree[key]))
    return words_acc


def get_lemmas(words):
    """
    Computes the lemmas of all words pulled from the email.
    :param words: The words (POS tagged) from the email.
    :return: A list of the words' lemmas.
    """
    digit_regex = re.compile(r'\d')

    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    lemmas = []
    for word in words:
        try:
            '''
            The words have been POS tagged and are a tuple in the following form:
            (word, POS TAG)
               0       1
            '''
            part_of_speech = POS_TO_WORDNET[word[1]]
            lemma = lemmatizer.lemmatize(word[0], pos=part_of_speech).lower()

            if lemma not in IRRELEVANT_WORDS and digit_regex.search(lemma) is None:
                lemmas.append(lemma)
        except KeyError:
            pass

    return lemmas


def get_similarity(email_word, tree_word):
    """
    Uses FastText to compute the similarity between two words.
    :param email_word: The word from the email to compare.
    :param tree_word: The word from the ontology tree to compare.
    :return: The similarity score of the two words.
    """
    try:
        sim_score = WORD_VECTORS.similarity(email_word, tree_word)
        return sim_score
    except KeyError:
        return 0


def extend_tree(current_tree, count_depth):
    """
    Finds hyponyms and uses them to extend the ontology tree.
    :param current_tree: The tree to be extended.
    :param count_depth: How many more rows of extensions to perform.
    :return: The extended tree.
    """
    for key in current_tree:
        if len(current_tree[key]) == 0 and count_depth > 0:
            synsets = wn.synsets(key)
            new_branches = {}
            for synset in synsets:
                hyponyms = synset.hyponyms()
                for hyponym in hyponyms:
                    lemmas = hyponym.lemmas()
                    for lemma in lemmas:
                        '''
                        The lemmatizer puts underscores between multiple word lemmas.
                        E.g. information_technology
                        These are split up and the latter words are put in the next row down in the tree.
                        '''
                        lemma_name = lemma.name()
                        if '_' in lemma_name:
                            multiple_words = lemma_name.split('_')
                            first_lemma = multiple_words[0]
                            remaining_lemmas = multiple_words[1:]

                            remaining_lemmas.reverse()
                            furthest_branch = {}
                            current_branch = {}
                            for word in remaining_lemmas:
                                current_branch = {word: furthest_branch}
                                furthest_branch = current_branch
                            new_branches[first_lemma] = current_branch
                        else:
                            new_branches[lemma_name] = {}

            current_tree[key] = extend_tree(new_branches, count_depth - 1)

    return current_tree


def classify_email(text, tree):
    """
    Classifies an email according to the ontology tree.
    :param text: The email text.
    :param tree: The tree to be used for classification.
    :return: The classification of the email.
    """
    common_words = read_common_words()
    words = retrieve_tags(text, common_words)

    lemmas = get_lemmas(words)
    lemmas.extend(check_tree(text, tree))

    # loop until the best tag is found
    classified = False
    tree_acc = []
    saved_sim_score = 0
    while not classified:
        # get the highest average similarity score of each node in the next level of the tree
        best_key = ""
        highest_sim_score = 0
        for key in tree:
            num_lemmas = len(lemmas)

            # Avoid ZeroDivisionError
            if num_lemmas > 0:
                sim_acc = 0
                for lemma in lemmas:
                    sim_acc += get_similarity(lemma, key)

                sim_avg = sim_acc / num_lemmas
            else:
                sim_avg = 0

            # if the average of this route is higher, then we plan to follow this route
            if sim_avg > highest_sim_score:
                best_key = key
                highest_sim_score = sim_avg

        # if the current tag has a higher score then we don't proceed
        if highest_sim_score > saved_sim_score:
            if best_key != "":
                tree = tree[best_key]
                if len(tree) == 0:
                    classified = True
                else:
                    tree_acc.append(best_key)
                    saved_sim_score = highest_sim_score
            else:
                classified = True
        else:
            classified = True

    return tree_acc


# extend the tree
extended_tree = extend_tree(TREE, 2)
pprint(extended_tree)

# loops though each email and appropriately classifies it
for file in TEST_FILES:
    file_path = os.path.join(TEST_PATH, file)
    with open(file_path) as email_file:
        file_text = email_file.read()
        classification = classify_email(file_text, extended_tree)

        print(file)
        print(str(classification))
