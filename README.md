# seminar-email-tagger
Python code using the NLTK library to perform Entity Tagging and Ontology tagging on emails about seminars.

## Entity Tagging/Information Extraction
The program performs Information Extraction on the emails, and then tags the information it has retrieved.

This has been evaluated on sample emails, with evaluate_information_extraction.py, to achieve the following results.

| Tag | Precision | Recall | F-Measure | Techniques |
| --- | --------- | ------ | --------- | ---------- |
| Start Time | 0.96 | 0.81 | 0.85 | - Regex on the email headers <br>- Relation Extraction on the email body|
| End Time | 0.96 | 0.82 | 0.85 | - Regex on the email headers <br>- Relation Extraction on the email body|
| Location | 0.91 | 0.67 | 0.68 | - Regex on header <br>- Relation Extraction on the body <br>- If not location has been found, search for previously found locations |
| Speaker | 0.82 | 0.48 | 0.48 | - Regex on header <br>- Relation Extraction on body <br>- Wikification to determine if a noun next to a speaker's name is a part of their name <br>- If a name hasn't been found, use the Stanford NER tagger and assume that a name on its own line is a speaker |
| Sentence | 0.8 | 0.7 | 0.67 | - NLTK sentence tokenizer <br>- If the sentence does not end in punctuation, then don't class it as a sentence <br>- Find the average length of a sentence in the training set, use this to create an upper and lower bound in which sentences should fit. |
| Paragraph | 0.79 | 0.58 | 0.59 | - Fit paragraphs around the tagged sentences. |

## Ontology Tagging
A small ontology tree must be created manually before running the program. I found the most popular words in the dataset I had, and used these to create the tree. <br>

The program will use wordnet to find hyponyms of the words in the tree, which it will use to expand the tree by two levels (any deeper and I found the classifications to be too specific). Some of the returned hyponyms are multiple words separated by underscores. These words are separated to be subtrees of each other. <br>

Each email is tokenized, and has stop words removed.<br>

The remaining words are then tagged with a POS tagger, and then lemmatized as the lemma is more likely to be in the tree.<br>

The Gensim library is used with pretrained word-vectors. The average similarity score is calculated between each subtree node and each lemma. The branch with the highest similarity score is then followed. However, if the current node has a higher similarity score then the email is classified as that node.
 
## To Run
- Put manually tagged, training, emails in /training/tagged. A sample email is provided
- Put test emails in /test/untagged, and manually tagged versions in /test/tagged. Sample emails are provided
- For Entity Tagging, run evaluate_information_extraction.py. The emails tagged by the program will be stored in /tagged
- For Ontology Tagging, update the manual Ontology Tree in ontology_tagging.py if required, and then run the file

