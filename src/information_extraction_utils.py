import string
import sys
import urllib

TAGS = ["<date>", "</date>", "<stime>", "</stime>", "<etime>", "</etime>", "<location>", "</location>", "<speaker>",
        "</speaker>", "<sentence>", "</sentence>"]


def format_file(file):
    """
    Splits an email into the header and body.
    :param file: The complete email
    :return: header: The email header.
    :return: body: The body (including any nested headers).
    """
    separated = file.split('\n\n', 1)
    if len(separated) == 2:
        header = separated[0]
        body = separated[1]
    elif len(separated) > 2:
        header = separated[0]
        body = ""
        for count in range(1, len(separated) - 1):
            body += separated[count]
    else:
        header = ""
        body = file

    return header, body


# wikification
# pulled from the lab file
def get_url(domain, url):
    # Headers are used if you need authentication
    headers = {}

    # If you know something might fail - ALWAYS place it in a try ... except
    try:
        conn = http.client.HTTPSConnection(domain)
        conn.request("GET", url, "", headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return data
    except Exception as e:
        # These are standard elements in every error.
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

    # Failed to get data!
    return None


# used to query wiki
# pulled from the lab file
def query_wiki(query):
    # If this is included as a module, you have access to the above function
    # without "running" the following piece of code.
    # Usually, you place functions in a file and their tests in chunks like below.
    if __name__ == '__main__':

        query = query  # "birmingham"#"obama"
        # This makes sure that any funny charecters (including spaces) in the query are
        # modified to a format that url's accept.
        query = urllib.parse.quote_plus(query)

        # Call our function.
        url_data = get_url('en.wikipedia.org', '/w/api.php?action=query&list=search&format=json&srsearch=' + query)

        # We know how our function fails - graceful exit if we have failed.
        if url_data is None:
            print("Failed to get data ... Can not proceed.")
            # Graceful exit.
            sys.exit()

        # http.client socket returns bytes - we convert this to utf-8
        url_data = url_data.decode("utf-8")

        return url_data
    return None


def detokenize(tokens):
    """
    Puts the tokens back into sentences.
    :param tokens: A list of tokens.
    :return: The tokens put back into a single string.
    """
    line = ""
    count_tokens = 0
    for token in tokens:
        if len(token) > 0:
            if token == "<sentence>":
                if tokens[count_tokens - 1] in string.punctuation:
                    line += " " + token
            if token in TAGS:
                line += token
            elif token[0] in string.punctuation:
                line += token
            elif count_tokens > 0:
                if tokens[count_tokens - 1] in TAGS and "/" not in tokens[count_tokens - 1]:
                    line += token
                else:
                    line += " " + token
            else:
                line += token
            count_tokens += 1
    return line
