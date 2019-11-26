from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
import information_extraction
import re
import os

TEST_TAGGED_PATH = '../test/tagged'
FILE_OUTPUT_PATH = '../tagged/'

TAGS = ["<stime>", "<etime>", "<location>", "<speaker>", "<sentence>", "<paragraph>"]
STIME_TAG = "<stime>"
ETIME_TAG = "<etime>"
LOCATION_TAG = "<location>"
SPEAKER_TAG = "<speaker>"
PARAGRAPH_TAG = "<paragraph>"
SENTENCE_TAG = "<sentence>"

TEST_FILES = [f for f in listdir(TEST_TAGGED_PATH) if isfile(join(TEST_TAGGED_PATH, f))]
NUM_FILES = len(TEST_FILES)


def remove_tags(text):
    """
    Remove tags from a tagged file to get the original text.
    :param text: The tagged text.
    :return: The plain text.
    """
    for tag in TAGS:
        text = re.sub(tag, "", text)
        '''
        Also need to remove the closing tags.
        E.g.   <  /  etime>
               0     1:
        '''
        text = re.sub(tag[0] + "/" + tag[1:], "", text)
    return text


def write_to_file(text, file_name):
    """
    Writes text to a file.
    :param text: The text to be written.
    :param file_name: The file to be written to.
    """
    file = open(FILE_OUTPUT_PATH + file_name, "w")
    file.write(text)
    file.close()


def calculate_precision(false_positives, num_tagged):
    """
    Calculates the precision of a specific tag.
    :param false_positives: The number of false positives of the tag.
    :param num_tagged: The number of times the tag was used.
    :return: The precision of the tag.
    """
    if false_positives == 0:
        precision = 1.0
    elif num_tagged == 0:
        precision = 0
    else:
        precision = (num_tagged - false_positives) / num_tagged

    return precision


def calculate_recall(num_tagged, false_positives, false_negatives):
    """
    Calculates the recall of a specific tag.
    :param num_tagged: The number of times the tag was used.
    :param false_positives: The number of false positives of the tag.
    :param false_negatives: The number of false negatives of the tag.
    :return: The recall of the tag.
    """
    denominator = num_tagged - false_positives + false_negatives
    if false_negatives == 0:
        recall = 1.0
    elif denominator == 0:
        recall = 0
    else:
        recall = (num_tagged - false_positives) / denominator

    return recall


def calculate_f_measure(precision, recall):
    """
    Calculates the f-measure of a specific tag.
    :param precision: The precision of the tag.
    :param recall: The recall of the tag.
    :return: The f-measure of the tag.
    """
    precision_plus_recall = precision + recall
    if precision_plus_recall == 0:
        f_measure = 0
    else:
        f_measure = 2 * ((precision * recall) / precision_plus_recall)

    return f_measure


def evaluate(my_tagged, test_tagged, tags_precision, tags_recall, tags_f_measure):
    """
    Compares the version tagged by information_extraction.py against the test dataset.
    :param my_tagged: The email tagged by information_extraction.py
    :param test_tagged: The tagged email from the test dataset.
    :param tags_precision: Accumulated precision scores for each tag over the set of emails.
    :param tags_recall: Accumulated recall scores for each tag over the set of emails.
    :param tags_f_measure: Accumulated f-measure scores for each tag over the set of emails.
    :return: The updated precision, recall, and f-measure cumulative scores.
    """

    for tag in TAGS:
        '''
        Construct the closing tag.
        E.g.   <  /  etime>
               0     1:
        '''
        end_tag = tag[0] + "/" + tag[1:]
        tag_regex = re.compile(rf'({tag})([\s\S]*?)({end_tag})')
        my_data = tag_regex.findall(my_tagged)
        test_data = tag_regex.findall(test_tagged)

        my_data_tags_removed = []
        for item in my_data:
            tags_removed = remove_tags(item[1])
            tokenized = word_tokenize(tags_removed)
            my_data_tags_removed.append(tokenized)

        test_data_tags_removed = []
        for item in test_data:
            tags_removed = remove_tags(item[1])
            tokenized = word_tokenize(tags_removed)
            test_data_tags_removed.append(tokenized)

        # things I tagged which shouldn't be tagged
        false_positives = [i for i in my_data_tags_removed + test_data_tags_removed if i not in test_data_tags_removed]
        # things I haven't tagged which should be tagged
        false_negatives = [i for i in my_data_tags_removed + test_data_tags_removed if i not in my_data_tags_removed]

        num_false_positives = len(false_positives)
        num_false_negatives = len(false_negatives)
        num_tags = len(my_data)

        precision = calculate_precision(num_false_positives, num_tags)
        recall = calculate_recall(num_tags, num_false_positives, num_false_negatives)
        f_measure = calculate_f_measure(precision, recall)

        tags_precision[tag] += precision
        tags_recall[tag] += recall
        tags_f_measure[tag] += f_measure

    return tags_precision, tags_recall, tags_f_measure


# train the tagger code
information_extraction.train_para_tagger()
information_extraction.train_location_tagger()

tags_precision = {STIME_TAG: 0,
                  ETIME_TAG: 0,
                  LOCATION_TAG: 0,
                  SPEAKER_TAG: 0,
                  PARAGRAPH_TAG: 0,
                  SENTENCE_TAG: 0}
tags_recall = {STIME_TAG: 0,
               ETIME_TAG: 0,
               LOCATION_TAG: 0,
               SPEAKER_TAG: 0,
               PARAGRAPH_TAG: 0,
               SENTENCE_TAG: 0}

tags_f_measure = {STIME_TAG: 0,
                  ETIME_TAG: 0,
                  LOCATION_TAG: 0,
                  SPEAKER_TAG: 0,
                  PARAGRAPH_TAG: 0,
                  SENTENCE_TAG: 0}

for file in TEST_FILES:
    print(file)

    file_path = os.path.join(TEST_TAGGED_PATH, file)
    with open(file_path) as email_file:
        file_text = email_file.read()
        detagged = remove_tags(file_text)
        my_tagged = information_extraction.tag_email(detagged)
        write_to_file(my_tagged, file)

        '''
        Response from evaluation:
        (tags_precision, tags_recall, tags_f_measure)
        '''
        evaluation = evaluate(my_tagged, file_text, tags_precision, tags_recall, tags_f_measure)
        tags_precision = evaluation[0]
        tags_recall = evaluation[1]
        tags_f_measure = evaluation[2]


# print average results across all files
for tag in TAGS:
    print(tag)
    print("Precision: " + str(round(tags_precision[tag] / NUM_FILES, 2)))
    print("Recall: " + str(round(tags_recall[tag] / NUM_FILES, 2)))
    print("F-Measure: " + str(round(tags_f_measure[tag] / NUM_FILES, 2)))
