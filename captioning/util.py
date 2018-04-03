import string
import numpy as np

punct_remover = str.maketrans('', '', string.punctuation)

def caption_to_words(caption):
    """Does some basic processing (removes punctuation, etc.), and splits."""
    caption = caption.lower()
    caption = caption.replace(". ", " EOS ")
    caption = caption.replace(".", " EOS ")
    caption = caption.translate(punct_remover)
    caption = caption.replace("EOS", "<EOS>")
    return caption.split()

def words_to_string(words): 
    """Inverse of caption to words, more or less."""
    outwords = []
    for i in range(len(words)):
        if words[i] == "<PAD>":
            continue
        elif words[i] == "<EOS>":
            outwords.append(".")
        else:
            outwords.append(words)

    return " ".join(outwords)

def load_vocabulary_to_index(filename):
    """Takes a vocab file with one word on each line, turns into a dict where
       vocab_mapping[word] = index, with indices in order of apperance in file.
    """
    vocab_mapping = {}
    backward_vocab = []
    with open(filename, "r") as v_file:
        for i, line in enumerate(v_file):
            word = line.rstrip()
            vocab_mapping[word] = i
            backward_vocab.append(word)

    return vocab_mapping, backward_vocab, i + 1 # last term is length of vocabulary

def words_to_indices(words, vocabulary, unk_token="<UNK>"):
    """Converts words to indices, handling unknowns.""" 
    return [vocabulary[word] if word in vocabulary else vocabulary[unk_token] for word in words]

def indices_to_words(indices, backward_vocabulary, unk_token="<UNK>"):
    """Converts words to indices, handling unknowns.""" 
    return [backward_vocabulary[index] for index in indices]

def pad_or_trim(words, length, right=True, pad_token="<PAD>"):
    """Pads or trims to fixed length, on left or right side, also returns mask."""
    curr_length = len(words)
    if curr_length > length:
        if right:
            words = words[:length]
            mask = np.ones([1, length])
        else:
            words = words[-length:]
            mask = np.ones([1, length])
    else:
        if right:
            words = words + ["<PAD>"] * (length-curr_length) 
            mask = np.concatenate([np.ones([1,curr_length], np.bool), np.zeros([1, (length-curr_length)], np.bool)], -1)
        else:
            words = ["<PAD>"] * (length-curr_length) + words 
            mask = np.concatenate([np.zeros([1, (length-curr_length)], np.bool), np.ones([1,curr_length], np.bool)], -1)
    
    return {"words": words, "mask": mask}
