from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger, StanfordNERTagger
from nltk.corpus import treebank, names
from os.path import isfile, join
from os import listdir
from information_extraction_utils import detokenize, format_file, query_wiki
import os
import nltk.data
import re
import string
from file_locations import TRAINING_CORPORA_PATH, STANFORD_TAGGER_PATH, STANFORD_TAGGER_DICTIONARY, JAVA_PATH

# constants
TAGS = ["<date>", "</date>", "<stime>", "</stime>", "<etime>", "</etime>", "<location>", "</location>", "<speaker>",
        "</speaker>", "<sentence>", "</sentence>"]
TITLES = ["mr", "mrs", "mr", "dr", "professor", "prof", "doctor", "md", "phd"]
STIME_TAG = "stime"
ETIME_TAG = "etime"
PARAGRAPH_TAG = "paragraph"
SENTENCE_TAG = "sentence"
LOCATION_TAG = "location"
SPEAKER_TAG = "speaker"
TRAIN_SENTS = treebank.tagged_sents()
UNIGRAM = UnigramTagger(TRAIN_SENTS, backoff=DefaultTagger('NN'))
BIGRAM = BigramTagger(TRAIN_SENTS, backoff=UNIGRAM)
TRIGRAM = TrigramTagger(TRAIN_SENTS, backoff=BIGRAM)
NAMES = names.words('male.txt') + names.words('female.txt') + names.words('family.txt')
os.environ['JAVAHOME'] = JAVA_PATH

# global variables
sentence_length_upper_bound = 0
sentence_length_lower_bound = 0
locations = set([])


def train_para_tagger():
    """
    Trains the paragraph tagger by calculating the average length of a sentence in the training set.
    :return:
    """
    global sentence_length_lower_bound
    global sentence_length_upper_bound
    training_files = [f for f in listdir(TRAINING_CORPORA_PATH) if isfile(join(TRAINING_CORPORA_PATH, f))]

    num_sentences = 0
    num_words = 0
    for file in training_files:
        file_text = open(TRAINING_CORPORA_PATH + "/" + file, "r").read()
        removed_first_tag = file_text.split("<paragraph>")
        first_tag_removed = []
        if len(removed_first_tag) > 1:
            first_tag_removed += removed_first_tag[1:]

        paragraphs = []
        for split in first_tag_removed:
            removed_second_tag = split.split("</paragraph>")
            if len(removed_second_tag) > 1:
                paragraphs += removed_second_tag[:len(removed_second_tag) - 1]
            else:
                paragraphs += removed_second_tag

        for paragraph in paragraphs:
            sentences = paragraph.split("<sentence>")
            for sentence in sentences:
                tag_removed = sentence.replace("</sentence>", "")
                for tag in TAGS:
                    tag_removed = tag_removed.replace(tag, "")

                tokenised_text = nltk.word_tokenize(tag_removed)
                words = tokenised_text
                for word in words:
                    all_punc = True
                    for char in word:
                        if char not in string.punctuation:
                            all_punc = False
                    # punctuation is not counted as a word
                    if not all_punc:
                        num_words += 1

                num_sentences += 1

    avg_sentence_length = num_words / num_sentences
    sentence_length_upper_bound = avg_sentence_length * 1.5
    sentence_length_lower_bound = avg_sentence_length * 0.5


def train_location_tagger():
    """
    Pulls all of the locations out of the training set.
    :return:
    """
    training_files = [f for f in listdir(TRAINING_CORPORA_PATH) if isfile(join(TRAINING_CORPORA_PATH, f))]
    global locations
    for file in training_files:
        file_text = open(TRAINING_CORPORA_PATH + "/" + file, "r").read()
        file_locations = re.findall(r'(<location>?)(.*)(</location>?)', file_text)
        for location in file_locations:
            location_name = location[1]
            for tag in TAGS:
                location_name = location_name.replace(tag, "")
                location_name = location_name.replace(tag[0] + "/" + tag[0:], "")
                locations.add(location_name)


def check_header(header, tags):
    """
    Uses regular expressions on the header to pull out relevant information.
    :param header: The email header.
    :param tags: The accumulated strings which have been tagged. E.g. [('dave', 'speaker')]
    :return: The updated tags list.
    """
    time = re.search(r'Times?:\s*(.*)\n', header, flags=re.I)
    place = re.search(r'Places?:\s*(.*)\n|Locations?:\s*(.*)\n', header, flags=re.I)
    speaker = re.search(r'Who:\s*([^,\n]*)(\n\s*(.*))?|Speakers?:\s*([^,\n]*)(\n\s*(.*))?', header, flags=re.I)
    if time is not None:
        time = time.group(1)
        start_and_end_time = re.search(r'(.*?)\s?([-,;]|until|up\sto)\s?(.*)', time)
        if start_and_end_time is not None:
            start_time = start_and_end_time.group(1)
            end_time = start_and_end_time.group(3)
            tags.add((start_time, STIME_TAG))
            tags.add((end_time, ETIME_TAG))
        else:
            tags.add((time, STIME_TAG))
    if place is not None:
        # Places:
        place_name = place.group(1)
        if place_name is None:
            # Locations:
            place_name = place.group(2)
        tags.add((place_name, LOCATION_TAG))
        # Add location to list of known locations
        locations.add(place_name)
    if speaker is not None:
        # Who:
        speaker_name = speaker.group(1)
        if speaker_name is None:
            # Speaker:
            speaker_name = speaker.group(4)
        # Remove new line
        speaker_name = speaker_name.strip()
        # Often a punctuation character at the end, which must be removed.
        if speaker_name[-1:] in string.punctuation:
            speaker_name = speaker_name[:-1].strip()
        tags.add((speaker_name, SPEAKER_TAG))
    return tags


def check_noun(word):
    """
    Decides whether a noun is likely to be a person or a place by using wikification.
    :param word: The noun to check.
    :return: Whether the noun is a place or person.
    """
    wiki_results = query_wiki(word)
    if wiki_results is not None:
        if "born" in wiki_results:
            return "person"
        elif word in NAMES:
            return "person"
        elif "founded" in wiki_results:
            return "place"
        else:
            return "place"
    else:
        # if there are no wiki results then we can assume they're a person
        return "person"


def rel_extract(body, tags):
    """
    Performs relation extraction on the body, to try and pull out relevant information.
    :param body: The email body.
    :param tags: A list of accumulated strings which have been tagged so far.
    :return: The updated list of accumulated tagged strings.
    """
    find_rels = re.search(
        r'((\w*\s*)?\w*)(\s*from\s*((the)?\s*university\s*of\s*\w*|\w*\s*university))?\s*(will|is\sgoing\sto)\s*('
        r'present|speak|talk|lecture|deliver\s*a\s*(guest\s*)?(lecture|talk|presentation)?)\s*('
        r'on\s*the\s*topic|in|about|on)\s*(.*)\.',
        body, flags=re.I)
    if find_rels is not None:
        speaker = find_rels.group(1)
        if check_noun(speaker) == "person":
            tags.add((speaker, SPEAKER_TAG))
    find_rels = re.search(
        r'((\w*\s*)?\w*)(\s*from\s*((the)?\s*university\s*of\s*\w*|\w*\s*university))?\s*(will|is\sgoing\sto)\s*('
        r'present|speak|talk|lecture|deliver\s*a\s*(guest\s*)?(lecture|talk|presentation)?)',
        body, flags=re.I)
    if find_rels is not None:
        speaker = find_rels.group(1)
        if check_noun(speaker) == "person":
            tags.add((speaker, SPEAKER_TAG))
    find_rels = re.search(
        r'.*\sThe\s(seminar|lecture|talk|presentation)\s(will|is going to)\s(be\s(held\s)?in|hosted in)\s(.*)\sat\s('
        r'.*)\son\s([^!\.]*)',
        body, flags=re.I)
    if find_rels is not None:
        location = find_rels.group(5)
        time = find_rels.group(6)
        tags.add((location, LOCATION_TAG))
        locations.add(location)
        tags.add((time, STIME_TAG))
    else:
        find_rels = re.search(
            r'.*\sThe\s(seminar|lecture|talk|presentation)\s(will|is going to)\s(be\s(held\s)?in|hosted in)\s('
            r'.*)\son\s(.*)\sat\s([^!\.]*)',
            body, flags=re.I)
        if find_rels is not None:
            location = find_rels.group(5)
            time = find_rels.group(7)
            tags.add((location, LOCATION_TAG))
            locations.add(location)
            tags.add((time, STIME_TAG))
        else:
            find_rels = re.search(r'.*\sThe\s(seminar|lecture|talk|presentation)\s(will|is going to)\s(be\s('
                                  r'held\s)?in|hosted in)\s(.*)\.', body, flags=re.I)
            if find_rels is not None:
                location = find_rels.group(5)
                tags.add((location, LOCATION_TAG))
                locations.add(location)
    find_rels = re.search(
        r'(seminar|talk|presentation).*(are|will\sbe|is\sgoing\sto\sbe).*at\s(\d*(\D\d*)?(\s(pm|am))?)', body,
        flags=re.I)
    if find_rels is not None:
        time = find_rels.group(3)
        tags.add((time, STIME_TAG))
    find_rels = re.search(r'(seminar|talk|presentation).*(are|will\sbe|is\sgoing\sto\sbe).*in\s(\w*(\s\d*))', body,
                          flags=re.I)
    if find_rels is not None:
        location = find_rels.group(3)
        tags.add((location, LOCATION_TAG))
        locations.add(location)
    return tags


def tag_sents_and_paras(text):
    """
    Tags the sentences and paragraphs in the body.
    :param text: The text to tag.
    :return: The list of tokens with paragraph and sentence tags added.
    """
    '''
    We assume each paragraph is split by an empty line.
    The last item will probably be empty, as there is often free lines at the end of the file, so it is removed
    '''
    paras = text.split("\n\n")[:-1]

    sents_tagged = []
    count_sents_tagged = 0
    for para in paras:
        tokenised = nltk.sent_tokenize(para)
        para_start_index = count_sents_tagged

        # Loop until enough sentences for the paragraph have been found.
        all_sents = True
        for token in tokenised:
            last_char = token[-1:]
            sent_length = len(token)
            if last_char == "." or last_char == "!" or last_char == "?" and sentence_length_lower_bound < sent_length < sentence_length_upper_bound:
                sents_tagged.append("<sentence>")
                sents_tagged.append(token[:-1])
                sents_tagged.append("</sentence>")
                sents_tagged.append(last_char)
                count_sents_tagged += 4
            else:
                all_sents = False
                sents_tagged.append(token)
                sents_tagged.append("\n")
                count_sents_tagged += 2

        if all_sents:
            sents_tagged.insert(para_start_index, "<paragraph>")
            sents_tagged.append("</paragraph>")
            sents_tagged.append("\n\n")
            count_sents_tagged += 3

    return sents_tagged


def tag_body(text, tags):
    """
    Searches the email body for text which has been identified as needing to be tagged, and adds the tags.
    :param text: The text to be tagged.
    :param tags: A list of strings and their associated tags to search for.
    :return: The tagged text.
    """
    for tag in tags:
        '''
        The tag is in the following format:
        (string, tag)
        '''
        tag_string = tag[0]
        tag_tag = tag[1]
        # Remove full stops from times as they definitely don't belong
        if (tag_tag == STIME_TAG or tag_tag == ETIME_TAG) and tag_string[-1:] == ".":
            tag_string = tag_string[:-1]

        # Find instances of the tagged string and tag it
        for index in re.finditer(re.escape(tag_string), text):
            before_substring = text[:index.start()]
            after_substring = text[index.end():]
            text = f'{before_substring} <{tag_tag}>{tag_string} </{tag_tag}>{after_substring}'

    return text


def tag_header(header, tags):
    """
    Searches the email header for text which has been identified as needing to be tagged, and adds the tags.
    :param header: The email header to be tagged.
    :param tags: A list of strings and their associated tags to search for.
    :return: The tagged header.
    """
    header_lines = header.splitlines()
    tagged = ""
    for line in header_lines:
        info_tagged = tag_body(line, tags)
        tagged += info_tagged + "\n"
    return tagged


def find_names(text, tags):
    """
    Finds names with a Stanford tagger. These are assumed to be a speaker name if no speaker has been found.
    :param text: The text to search.
    :param tags: A list of accumulated strings which have been tagged so far.
    :return: The updated list of accumulated tags.
    """
    stanford_tagger = StanfordNERTagger(STANFORD_TAGGER_DICTIONARY,
                                        STANFORD_TAGGER_PATH,
                                        encoding='utf-8')
    tokenised = nltk.word_tokenize(text)
    classified = stanford_tagger.tag(tokenised)

    names = []
    current_name = ""
    found_name = False
    for token in classified:
        '''
        The tag is in the following format:
        (string, tag)
        '''
        token_word = token[0]
        token_tag = token[1]
        is_shortened_name = re.match("^\w+\.$", token_word)
        if token_tag == "PERSON" or token_word in NAMES or token_word.lower() in TITLES or is_shortened_name is not None or token_word in string.punctuation:
            if found_name is True and token_word not in string.punctuation:
                current_name += " " + token_word
            else:
                current_name += token_word
            found_name = True
        elif token_tag != "PERSON" and found_name:
            names.append(current_name)
            current_name = ""
            found_name = False

    lines = text.splitlines()
    for line in lines:
        stripped_line = line.strip()
        if stripped_line in names:
            tags.add((line, SPEAKER_TAG))
            break

    return tags


def find_locations(text, tags):
    """
    Searches the text for previously identified locations, if no other location has been found.
    :param text: The text to search.
    :param tags: A list of accumulated strings which have been tagged so far.
    :return: The updated list of accumulated tags.
    """
    lower_text = text.lower()
    for location in locations:
        if location.lower() in lower_text:
            tags.add((location, LOCATION_TAG))

    return tags


def tag_email(email_text):
    """
    Performs information extraction on an email and adds tags.
    :param email_text: The email to tag.
    :return: The tagged email.
    """
    '''
    Split the email into header and body.
    return value is: (header, body)
    '''
    formatted = format_file(email_text)
    header = formatted[0]
    body = formatted[1]

    tags = set([])
    tags = check_header(header, tags)

    # A lot of emails have nested headers
    tags = check_header(body, tags)

    sent_and_para_tagged = tag_sents_and_paras(body)

    tags = rel_extract(body, tags)

    # if we haven't found either a speaker or a location we can fall back on the backup-methods
    speaker_tagged = False
    location_tagged = False
    for tag in tags:
        '''
        The tag is in the following format:
        (string, tag)
        '''
        tag_tag = tag[1]
        if tag_tag == SPEAKER_TAG:
            speaker_tagged = True
        elif tag_tag == LOCATION_TAG:
            location_tagged = True

    if not speaker_tagged:
        tags = find_names(body, tags)
        tags = find_names(header, tags)
    if not location_tagged:
        tags = find_locations(body, tags)
        tags = find_locations(header, tags)

    # tag all the information
    info_tagged = tag_body(detokenize(sent_and_para_tagged), tags)

    # put the tagged body back together
    detokenised = ""
    para_tokens = 0
    count_tokens = 0
    for token in info_tagged:
        if token == "<sentence>":
            previous_char = info_tagged[count_tokens - 1]
            if previous_char == "." or previous_char == "!" or previous_char == "?":
                detokenised += " " + token
            else:
                detokenised += "<paragraph>"
                detokenised += token
        elif token == "</sentence>":
            if info_tagged[count_tokens + 2] == "<sentence>":
                detokenised += token
            else:
                detokenised += token
                detokenised += info_tagged[count_tokens + 1]
                detokenised += "</paragraph>"
                detokenised += "\n\n"
                para_tokens = 2
        elif para_tokens > 0:
            detokenised += token
            para_tokens -= 1
        else:
            detokenised += token
        count_tokens += 1

    tagged_header = tag_header(header, tags)

    # put the email back together
    together = tagged_header + detokenised

    return together
