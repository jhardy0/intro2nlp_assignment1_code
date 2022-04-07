# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
from collections import Counter
import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')

#%%
# Analyse file

file_loc = 'data/preprocessed/train/sentences.txt'

with open(file_loc, 'r') as f:
    file = f.read()
# perform file operations

doc = nlp(file)

#%%
# Word frequencies

word_frequencies = Counter()

for sentence in doc.sents:
    words = []
    for token in sentence:
        # Let's filter out punctuation
        if not token.is_punct:
            words.append(token.text)
    word_frequencies.update(words)

#%%
num_tokens = len(doc)
num_words = sum(word_frequencies.values())
num_types = len(word_frequencies.keys())

print(f'num_tokens: \t {num_tokens}')
print(f'num_words: \t\t {num_words}')
print(f'num_types: \t\t {num_types}')
#%%
# Derive average sentence length and average word length
sentence_lengths = []

for sentence in doc.sents:
    words = []

    for token in sentence:
        # Let's filter out punctuation
        if not token.is_punct:
            words.append(token)

    if len(words) > 0:
        sentence_lengths.append(len(words))

average_sentence_length = np.round(np.mean(sentence_lengths),2)
print(f'sentence; avg length: \t {average_sentence_length}')

average_word_length = np.round(np.mean([len(word) for word in word_frequencies.keys()]),2)
print(f'word; avg length: \t {average_word_length}')

#%%

tag_dict = {}

for token in doc:
    tag = token.tag_

    if tag in tag_dict:
        tag_dict[tag] += 1
    else:
        tag_dict[tag] = 1

ordered_count_tags = list(reversed(sorted((tag_dict[i], i) for i in tag_dict)))

counts = [ordered_count_tags[i][0] for i in range(len(ordered_count_tags))]

rel_freq = [n / np.sum(counts) for n in counts]
print(np.round(rel_freq[:10],2))

most_occurring_tags = [ordered_count_tags[i][1] for i in range(10)]

most_occurring_tags

#%%

for i in range(10):
    print(f'{most_occurring_tags[i]} \t {spacy.explain(ordered_count_tags[i][1])}')
#%%

#%%

information_dict = {tag: {"token_freq": {}} for tag in most_occurring_tags}

i = 0
for token in doc:
    tag = token.tag_

    if tag in most_occurring_tags:
        tag_dict = information_dict[tag]["token_freq"]

        # break
        if token.text in tag_dict:
            information_dict[tag]["token_freq"][token.text] += 1
        else:
            information_dict[tag]["token_freq"][token.text] = 1

for tag in most_occurring_tags:
    tokens = information_dict[tag]["token_freq"]

    top_tokens = list(reversed(sorted((tokens[i], i) for i in tokens)))

    print(f'\ntop: \t {top_tokens[:3]}')
    print(f'inferior: \t {top_tokens[-1]}')

    # print(list(reversed(sorted((top_tokens, i) for i in top_tokens)))[:3])
    # break


#%%

def most_frequent_n_grams(sentences, n, top=3, POS=False):
    n_gram_counter = {}
    sentences = [s for s in sentences if not s.is_punct and not s.is_space] if POS else [s for s in sentences]

    n_grams = [sentences[i:i+n] for i in range(len(sentences)-n+1)]
    n_grams = [' '.join(str(e) for e in pair) for pair in n_grams]

    print(n_grams)
    for n_gram in n_grams:
        if n_gram not in n_gram_counter:
            n_gram_counter[n_gram] = 1
        else:
            n_gram_counter[n_gram] += 1
    top_n_grams = list(reversed(sorted((n_gram_counter[i], i) for i in n_gram_counter)))
    return top_n_grams[:top]

bigrams = most_frequent_n_grams(doc, n=2)
trigrams = most_frequent_n_grams(doc, n=3)

bigrams_POS = most_frequent_n_grams(doc, n=2, POS=True)
trigrams_POS = most_frequent_n_grams(doc, n=3, POS=True)

print(f'Bigram tokens: {bigrams}\n')
print(f'Trigram tokens: {trigrams}\n')

print(f'Bigram POS: {bigrams_POS}\n')
print(f'Trigram POS: {trigrams_POS}\n')



#%%
