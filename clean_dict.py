import re


def extract_words(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.readlines()

    words = set()

    for line in text:
        match = re.match(r'^([a-zA-Z][a-zA-Z\s\-\']*)\s{2,}|\t', line)
        if match:
            word = match.group(1).strip().lower()
            word = word.replace('-', ' ').replace("'", "")
            words.update(word.split())  # Add all individual words from the phrase to the set

    words = list(words)
    for word in words:
        if len(word) == 1:
            words.remove(word)

    unique_words = sorted(words)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(unique_words))


extract_words('oxford_old.txt', 'oxford.txt')