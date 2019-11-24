from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
import information_extraction
import re
import os
from file_locations import TRAINING_CORPORA_PATH, TEST_TAGGED_PATH, TEST_UNTAGGED_PATH, FILE_OUTPUT_PATH


TAGS = ["<stime>", "<etime>", "<location>", "<speaker>", "<sentence>", "<paragraph>"]
STIME_TAG = "<stime>"
ETIME_TAG = "<etime>"
LOCATION_TAG = "<location>"
SPEAKER_TAG = "<speaker>"
PARAGRAPH_TAG = "<paragraph>"
SENTENCE_TAG = "<sentence>"


# removes tags from a tagged file
def remove_tags(text):
    for tag in TAGS:
        text = re.sub(tag, "", text)
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


# compares my tagged version against the test tagged version to work out the statistics
def evaluate(my_tagged, test_tagged, tags_precision, tags_recall, tags_f_measure):
    num_tagged = {STIME_TAG: 0,
                  ETIME_TAG: 0,
                  LOCATION_TAG: 0,
                  SPEAKER_TAG: 0,
                  PARAGRAPH_TAG: 0,
                  SENTENCE_TAG: 0}

    false_positive_tags = {STIME_TAG: 0,
                           ETIME_TAG: 0,
                           LOCATION_TAG: 0,
                           SPEAKER_TAG: 0,
                           PARAGRAPH_TAG: 0,
                           SENTENCE_TAG: 0}

    false_negative_tags = {STIME_TAG: 0,
                           ETIME_TAG: 0,
                           LOCATION_TAG: 0,
                           SPEAKER_TAG: 0,
                           PARAGRAPH_TAG: 0,
                           SENTENCE_TAG: 0}

    # work out stats for each tag
    for tag in TAGS:
        end_tag = tag[0] + "/" + tag[1:]
        regex = "(" + tag + ")" + '([\s\S]*?)(' + end_tag + ')'
        my_data = re.findall(regex, my_tagged)
        test_data = re.findall(regex, test_tagged)
        my_data_tags_removed = []
        for item in my_data:
            tags_removed = remove_tags(item[1])
            tokenised = word_tokenize(tags_removed)
            my_data_tags_removed.append(tokenised)
            num_tagged[tag] += 1

        test_data_tags_removed = []
        for item in test_data:
            tags_removed = remove_tags(item[1])
            tokenised = word_tokenize(tags_removed)
            test_data_tags_removed.append(tokenised)

        # things I tagged which shouldn't be tagged
        false_positives = [i for i in my_data_tags_removed + test_data_tags_removed if i not in test_data_tags_removed]
        # things I haven't tagged which should be tagged
        false_negatives = [i for i in my_data_tags_removed + test_data_tags_removed if i not in my_data_tags_removed]

        false_positive_tags[tag] += len(false_positives)
        false_negative_tags[tag] += len(false_negatives)

    for tag in TAGS:
        precision = 0
        recall = 0
        f_measure = 0
        try:
            precision = (num_tagged[tag] - false_positive_tags[tag]) / num_tagged[tag]
        except:
            if (false_positive_tags[tag] == 0):
                precision = 1.0
            else:
                precision = 0
        try:
            recall = (num_tagged[tag] - false_positive_tags[tag]) / (
                        num_tagged[tag] - false_positive_tags[tag] + false_negative_tags[tag])
        except:
            if (false_negative_tags[tag] == 0):
                recall = 1.0
            else:
                recall = 0
        try:
            f_measure = 2 * ((precision * recall) / (precision + recall))
        except:
            f_measure = 0
        tags_precision[tag] += precision
        tags_recall[tag] += recall
        tags_f_measure[tag] += f_measure

    # return updated values
    return tags_precision, tags_recall, tags_f_measure


test_files = [f for f in listdir(TEST_TAGGED_PATH) if isfile(join(TEST_TAGGED_PATH, f))]

# train the tagger code
information_extraction.train_para_tagger()
information_extraction.train_location_tagger()

# set the default values
total_precision = 0
total_recall = 0
total_f_measure = 0
num_files = 0

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

# tag each file
for file in test_files:
    file_name = str(file)
    print(file_name)
    file_path = os.path.join(TEST_TAGGED_PATH, file)
    file_text = open(file_path, "r").read()
    detagged = remove_tags(file_text)
    my_tagged = information_extraction.tag_email(detagged)
    write_to_file(my_tagged, file_name)

    evaluation = evaluate(my_tagged, file_text, tags_precision, tags_recall, tags_f_measure)
    tags_precision = evaluation[0]
    tags_recall = evaluation[1]
    tags_f_measure = evaluation[2]
    num_files += 1

# print average results across all files
for tag in TAGS:
    print(tag)
    print("Precision: " + str(round(tags_precision[tag] / num_files, 2)))
    print("Recall: " + str(round(tags_recall[tag] / num_files, 2)))
    print("F-Measure: " + str(round(tags_f_measure[tag] / num_files, 2)))
