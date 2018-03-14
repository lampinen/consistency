import string

punct_remover = str.maketrans('', '', string.punctuation)

def caption_to_words(caption):
    """Does some basic processing (removes punctuation, etc.), and splits."""
    caption = caption.lower()
    caption = caption.replace(". ", " EOS ")
    caption = caption.replace(".", " EOS ")
    caption = caption.translate(punct_remover)
    caption = caption.replace("EOS", "<EOS>")
    return caption.split()

def load_vocabulary_to_index(filename):
    """Takes a vocab file with one word on each line, turns into a dict where
       vocab_mapping[word] = index, with indices in order of apperance in file.
    """
    vocab_mapping = {}
    with open(filename, "r") as v_file:
        for i, line in enumerate(v_file):
            vocab_mapping[line] = i
    return vocab_mapping

def words_to_indices(words, vocabulary, unk_token="<UNK>"):
    """Converts words to indices, handling unknowns.""" 
    return [vocabulary[word] if word in vocabulary else vocabulary[unk_token] for word in words]

def pad_or_trim(words, length, right=True, pad_token="<PAD>"):
    """Pads or trims to fixed length, on left or right side."""
    curr_length = len(words)
    if curr_length > length:
        if right:
            words = words[:length]
        else:
            words = words[-length:]
    else:
        if right:
            words = words + ["<PAD>"] * (curr_length-length) 
        else:
            words = ["<PAD>"] * (curr_length-length) + words 
    return words
