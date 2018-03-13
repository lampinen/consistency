import string

punct_remover = str.maketrans('', '', string.punctuation)

def caption_to_words(caption):
    caption = caption.lower()
    caption = caption.replace(". ", " EOS ")
    caption = caption.replace(".", " EOS ")
    caption = caption.translate(punct_remover)
    caption = caption.replace("EOS", "<EOS>")
    return caption.split()
