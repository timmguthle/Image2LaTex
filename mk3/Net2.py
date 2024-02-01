import keras
from keras import layers, Model
import tensorflow as tf

# disable GPU for now to avoid memory errors
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Encoder_CNN(keras.layers.Layer):
    '''
    Input: (batch_size, ImH, ImW, channles)
    Output: (batch_size, h, w, 512)
    '''


    def __init__(self, units=256, **kwargs):
        super().__init__(**kwargs)
        self.cnn_out_channels = int(units * 2)
        self.cnn3 = keras.layers.Conv2D(self.cnn_out_channels, (3,3), activation='relu', padding='same', name='cnn3')
        self.cnn4 = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same', name='cnn4')
        self.cnn5 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', name='cnn5')
        self.pool3 = keras.layers.MaxPool2D((2,2), strides=(2,2), name='pool3')
        self.cnn6 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='cnn6')
        self.pool4 = keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same', name='pool4')
        self.batch_norm = keras.layers.BatchNormalization(name='batch_norm1')
        

    def call(self, input):
        x = self.cnn6(input)
        x = self.pool4(x)
        x = self.cnn5(x)
        x = self.pool3(x)
        x = self.cnn4(x)
        x = self.cnn3(x)
        x = self.batch_norm(x)
        return x


class Row_Encoder(keras.layers.Layer):
    '''
    call this Layer on each row of the generated feature map after the cnn
    '''

    def __init__(self, units=256, batch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.lstm = keras.layers.LSTM(units, return_sequences=True, return_state=False, name='lstm1')
        #self.gru = keras.layers.GRU(units, return_sequences=True, return_state=False, name='gru1', recurrent_initializer='glorot_uniform')
        self.bi = keras.layers.Bidirectional(self.lstm, name='bi1', merge_mode='concat')
        self.batch_size_known = True
        self.initial_state = None
        self.units = units
        self.batch_size = batch_size

    def build(self, input_shape):

        # initialize the the initial states as weights for the RNN
        self.w1 = self.add_weight(shape=(self.batch_size, self.units), initializer='glorot_uniform' , trainable=True, name='w1')
        self.w2 = self.add_weight(shape=(self.batch_size, self.units), initializer='glorot_uniform' , trainable=True, name='w2')
        self.w3 = self.add_weight(shape=(self.batch_size, self.units), trainable=True,name='w3')
        self.w4 = self.add_weight(shape=(self.batch_size, self.units), trainable=True, name='w4')
        self.initial_state = [self.w1, self.w2, self.w3, self.w4]


    def call(self, inputs):
        x = self.bi(inputs, initial_state=self.initial_state)
        return x
        

        
class All_Row_Encoder(keras.layers.Layer):
    '''
    Input shape (None, h, w, 512)
    Output shape (None, (h*w), 512)

    for each row the RNN is executed, then the outputs are concat togehter. 
    '''

    def __init__(self, units=256, **kwargs):
        super().__init__(**kwargs)
        self.enc = Row_Encoder(name='row_encoder1', units=units)
        self.units = units
        self.all_row_units = int(units * 2) # for concat of the two directions of the bidirectional RNN

    @tf.function
    def call(self, x):
        first_pass = True
        new_tensor = tf.zeros((1,1,self.all_row_units))
        while tf.size(x) != 0:
            # initialize the tensorflow loop to generate a graph
            tf.autograph.experimental.set_loop_options(shape_invariants=((new_tensor, tf.TensorShape([None, None, self.all_row_units])), (x, tf.TensorShape([None, None, None, self.all_row_units]))))
            input = x[:,0,:,:] # always take the first row

            # call the Row_Encoder on the row
            output = self.enc(input)
            if first_pass:
                new_tensor = output
                first_pass = False
            else:
                new_tensor = tf.concat((new_tensor, output), axis=1) # flatten the tensor already here

            x = x[:,1:,:,:] # remove the first row

        return new_tensor


class ConcatLayer(keras.layers.Layer):
    '''
    A layer to flatten the output after the CNN. Can be used instead of the All_Row_Encoder

    just for debugging purposes, the final net should use the All_Row_Encoder
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reshape = keras.layers.Reshape((-1, 512), name='reshape1')

    def call(self, x):
        x = self.reshape(x)
        return x
  
    
class My_Attention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.additiv_attention = keras.layers.AdditiveAttention(name='additiv_attention1')
        self.mha = keras.layers.MultiHeadAttention(num_heads=1, key_dim=32, name='mha1')
        self.add = keras.layers.Add(name='add1')
        self.dot = keras.layers.Multiply(name='dot1')
        self.layernorm = keras.layers.LayerNormalization(name='layernorm1')
        self.attention_scores = 0

    def call(self, x, context):
        '''
        x: Query from the RNN decoder called h_t in the paper
        context: feature map from the output of the Row encoder called V tild _{h,w}

        important: flatten Feature map first (done by the all row encoder): (batch, h, w, channels) -> (batch, (h*w), channles)

        output is the context vector c_t concat with h_t

        input shape: (batch, (h*w), channels)
        output shape: (batch, token_length, channels)

        '''
        output, scores = self.mha(query=x, value=context, return_attention_scores=True)
        self.attention_scores = scores
        # maybe save scores in the future for ploting

        # it is possible to use self.add or concat instead of self.dot
        x = self.add([x, output])
        x = self.layernorm(x)

        return x


class Decoder(keras.layers.Layer):
    '''
    Recieves the context from the All_Row_Encoder and the tokens. 

    input shape: (batch, (h*w), channels) and (batch, token_length)
    output shape: (batch, token_length, vocab_size)
    
    returns logits for the next token on each input token. 
    '''
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    
    def __init__(self, text_processer, dec_units=512, **kwargs):
        super().__init__(**kwargs)
        # all the language processing stuff: 
        self.output_dim = 80 # parameter from the paper
        self.dec_units = dec_units
        self.text_processer = text_processer
        self.vocab_size = text_processer.latex_vocab_size
        self.word_to_id = text_processer.word_to_id
        self.id_to_word = text_processer.id_to_word
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')
        # Layers: 

        # convert ID to embedded vector
        self.embedding = keras.layers.Embedding(self.vocab_size, self.output_dim, mask_zero=True, name='embedding1') # mask_zero meight be set to False

        # RNN to process the embadded vectors
        # this corospondes to the h_t in the paper
        #self.rnn = keras.layers.LSTM(512, return_sequences=True, return_state=True)
        self.gru = keras.layers.GRU(512, return_sequences=True, return_state=True, name='gru2')

        # Attention Layer. Attention returns the context vector c_t added(or concat) to h_t
        self.attention = My_Attention(name='attention2')

        # generate logits 
        self.out_layer = keras.layers.Dense(self.vocab_size, name='out_layer1')


    def call(self, context, x, state=None, return_state=False):
        '''
        for generation pass in the final state of the run before as initial state
        '''
        x = self.embedding(x)

        # if given, set initial state of the decoder RNN
        x, state = self.gru(x, initial_state=state) # type: ignore

        # call the Attention Layer
        x = self.attention(x, context)

        logits = self.out_layer(x)

        # return either just the logits or the state and logits
        if return_state:
            return logits, state
        else:
            return logits
        
    def return_processed_text(self, text):
        '''
        for debugging purposes. returns the processed text after the embedding layer and the RNN
        '''
        x = self.embedding(text)
        x, _ = self.gru(x) # type: ignore
        return x

    def return_after_attention(self, context, x, state=None):
        '''
        for debugging purposes. returns the output of the attention layer
        '''
        x = self.embedding(x)
        x, _ = self.gru(x) # type: ignore
        x = self.attention(x, context)
        scores = self.attention.attention_scores
        return x, scores

    def get_initial_state(self, input):
        '''
        function for token generation. returns the first token, the initial state and a boolean vector to keep track of the end token
        '''
        start_tokens = tf.fill([tf.shape(input)[0], 1], self.start_token) # type: ignore
        is_done = tf.zeros([tf.shape(input)[0], 1], dtype=tf.bool) # type: ignore
        embedded_start_tokens = self.embedding(start_tokens)
        return start_tokens, is_done, self.gru.get_initial_state(embedded_start_tokens)
    

    def tokens_to_text(self, input):
        text = tf.strings.reduce_join(self.id_to_word(input), axis=-1, separator=' ')
        return text
    

    def get_next_token(self, context, token, is_done, state):
        '''
        for generation. returns the next token, the new state and the updated is_done vector
        '''
        logits, state = self(context, token, state=state, return_state=True) # type: ignore
        next_token = tf.argmax(logits, axis=-1)

        # if the next token is the end token, set is_done to True
        is_done = tf.logical_or(is_done, next_token == self.end_token)
        # if is_done is True, set next_token to 0
        next_token = tf.where(is_done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, is_done, state
   

class Decompiler(keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun
    
    def __init__(self, text_processer, units=256, dec_units=512, drop_out_value=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CNN_Encoder = Encoder_CNN(units=units, name='encoder')
        self.Row_Encoder = All_Row_Encoder(units=units, name='row_encoder')
        self.Decoder = Decoder(text_processer, dec_units=dec_units, name='my_decoder')
        self.dropout = keras.layers.Dropout(drop_out_value, name='dropout1')
        self.con = ConcatLayer(name='concat1')


    def call(self, inputs):
        features, x = inputs
        features = self.CNN_Encoder(features)

        # for debugging purposes leave out the Row_Encoder
        context = self.Row_Encoder(features)
        #context = self.con(features)
        context = self.dropout(context)

        logits = self.Decoder(context, x)
        return logits
    
    def return_context(self, image):
        '''
        for debugging purposes. returns the context vector after the Row_Encoder and the CNN_Encoder
        '''
        features = self.CNN_Encoder(image)
        context = self.Row_Encoder(features)
        return context, features

    
    def decompile(self, image, max_len=100):
        '''
        function to decompile an image

        image: image to decompile
        max_len: maximum length of the output formula

        returns: the decompiled formula as a string
        '''
        print(image.numpy().max(), image.numpy().min())
        context = self.CNN_Encoder(image)
        #context = self.Row_Encoder(context)
        context = self.con(context)

        tokens = []
        token, is_done, state = self.Decoder.get_initial_state(image)

        for _ in range(max_len):
            token, is_done, state = self.Decoder.get_next_token(context, token, is_done, state)
            tokens.append(token)

        tokens = tf.concat(tokens, axis=-1)

        result = self.Decoder.tokens_to_text(tokens)
        return result
    
if __name__ == '__main__':

    # debugging stuff
    input = keras.layers.Input(shape=(50, 200, 512))
    formula_input = keras.layers.Input(shape=(150))
    x = All_Row_Encoder()(input)
    model = keras.Model(inputs=input, outputs=x)
    model.summary()

