import tensorflow as tf
import keras


class TextProcesser:
    '''
    TextProcesser class is used to process the latex formulae into a format that can be used by the model.
    '''

    def __init__(self, vocab_path='/home/stud/ge42nog/projects/pix2tex/data/latex_vocab_new'):
        self.vocab_path = vocab_path
        self.latex_vocab, self.latex_vocab_size = self.load_vocab()
        self.word_to_id = keras.layers.StringLookup(vocabulary=self.latex_vocab, mask_token='', oov_token='[UNK]', name='word_to_id12')
        self.id_to_word = keras.layers.StringLookup(vocabulary=self.latex_vocab, mask_token='', oov_token='[UNK]', invert=True, name='id_to_word12')
        self.word_to_id_int = lambda x: int(word_to_id(x)) #type: ignore


    def load_vocab(self):
        # Tokenize the formulae
        with open(self.vocab_path, 'r') as f:
            latex_vocab = [word[:-1] for word in f.readlines()][:-1]

        latex_vocab.insert(0,'[START]')
        latex_vocab.insert(0, '[END]')
        latex_vocab.insert(0, '[UNK]')
        latex_vocab.insert(0, '')
        latex_vocab_size = len(latex_vocab)

        return latex_vocab, latex_vocab_size
    
    def Id_tensor_to_string(self, input_tensor):
        text = tf.strings.reduce_join(self.id_to_word(input_tensor), axis=-1, separator=' ')
        return text.numpy().decode('utf-8') # return type is string


