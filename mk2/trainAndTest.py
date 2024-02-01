import keras
import os
from keras import layers, Model
import tensorflow as tf
from Net2 import *
from PIL import Image
import numpy as np
from TextProcesser import TextProcesser
import matplotlib.pyplot as plt


# disable GPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# check for GPU access
physical_devices = tf.config.list_physical_devices('GPU')
print(f'GPU access: {physical_devices}')

# Define constants
BUFFER_SIZE = 1000
BATCH_SIZE = 16
MAX_TOKEN = 152
EPOCHS = 20

# replace with path to your processed data
DATA_DIR = '/home/stud/ge42nog/projects/processed_data/formula_images_processed/'
CHECKPOINT_PATH = '/home/stud/ge42nog/projects/im2latex_public/Net2/save1'

# load data
train = np.load('/home/stud/ge42nog/projects/im2latex_public/train_buckets.npy', allow_pickle=True)[0]
test = np.load('/home/stud/ge42nog/projects/im2latex_public/test_buckets.npy', allow_pickle=True)[0]
validate = np.load('/home/stud/ge42nog/projects/im2latex_public/validate_buckets.npy', allow_pickle=True)[0]


def load_im_from_filename(filename, size):
    '''
    input: path to file, size: relevant bucket size

    returns: normalized image file in float32 dtype
    
    '''
    raw = tf.io.read_file(filename)
    img = tf.io.decode_png(raw, 1) # type: ignore
    img = tf.cast(img, tf.float32)
    img = keras.layers.Rescaling(1./255)(img)
    img = tf.reshape(img, (size[1], size[0], 1))
    return img


def process_data(img, target):
    targ_in = target[:,:-1]
    targ_out = target[:,1:]
    return (img, targ_in), targ_out


def build_dataset(buckets, size):

    file_list = [(DATA_DIR + buckets[size][i][0]) for i in range(len(buckets[size]))]
    label_list = [np.array(buckets[size][i][1]) for i in range(len(buckets[size]))]
    for label in label_list:
        label.resize(MAX_TOKEN, refcheck=False) # padd labeles with zeroes to max length. inplace
        
    dataset_images = tf.data.Dataset.from_tensor_slices(file_list)
    dataset_images = dataset_images.map(lambda x: load_im_from_filename(x, size))
    dataset_labels = tf.data.Dataset.from_tensor_slices(label_list)

    dataset = tf.data.Dataset.zip(dataset_images, dataset_labels).batch(BATCH_SIZE, drop_remainder=True)

    dataset = dataset.map(process_data)

    return dataset


def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask
    # Return the total.
    final_loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return final_loss

def masked_acc(y_true, y_pred):
    # Calculate the accuracy for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)



# train dataset
train_ds_list = [build_dataset(train, size) for size in train.keys()]
# test dataset
test_ds_list = [build_dataset(test, size) for size in test.keys()]
# validation dataset
validate_ds_list = [build_dataset(validate, size) for size in validate.keys()]

# build complete datasets
complete_train_ds = train_ds_list[0]
for ds in train_ds_list[1:]:
    complete_train_ds = complete_train_ds.concatenate(ds)

complete_test_ds = test_ds_list[0]
for ds in test_ds_list[1:]:
    complete_test_ds = complete_test_ds.concatenate(ds)

complete_validate_ds = validate_ds_list[0]
for ds in validate_ds_list[1:]:
    complete_validate_ds = complete_validate_ds.concatenate(ds)

my_text_processor = TextProcesser()

model = Decompiler(my_text_processor, name='decompiler')

model.compile(
    optimizer=keras.optimizers.Adam(clipnorm=1.),
    loss=masked_loss,
    metrics=[masked_loss, masked_acc],
)

def start_training():
    save_callback = keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True, save_best_only=True, monitor='val_masked_acc', mode='max')
    eval_list = []
    for e in range(EPOCHS):
        complete_test_ds.shuffle(BUFFER_SIZE)
        model.fit(complete_train_ds.shard(4,0), epochs=1)
        model.fit(complete_train_ds.shard(4,1), epochs=1)
        model.fit(complete_train_ds.shard(4,2), epochs=1)
        model.fit(complete_train_ds.shard(4,3), epochs=1, validation_data=complete_validate_ds, callbacks=[save_callback])
        eval_list.append(model.evaluate(complete_validate_ds))
        print('Epoch: ', e, 'done')

    np.save('Net2/eval_list.npy', np.array(eval_list))

def simple_training():
    save_callback = keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True, save_best_only=True, monitor='val_masked_acc', mode='max')
    model.fit(complete_train_ds, epochs=20, validation_data=complete_validate_ds, callbacks=[save_callback])


def test_model():
    #model.load_weights('/home/stud/ge42nog/projects/im2latex_public/Net1/save3')
    complete_validate_ds.shuffle(BUFFER_SIZE)
    for (i,j),k in complete_validate_ds.take(2):
        input_image = tf.expand_dims(i[2], 0)
        pred = model.decompile(input_image)
        plt.imshow(input_image[0,:,:,:], cmap='gray_r')
        plt.show()
        final_string = [x.decode('utf-8') for x in pred[0].numpy()]
        print(final_string) 
        print(k[2]) 


def test_dataset():
    for (i,j), k in complete_train_ds.take(10):
        print(i.shape, j.shape, k.shape)
        #x = model((i,j))
        print(i.numpy().max())
        print(x.shape)


def debugging():
    complete_validate_ds.skip(3000)
    for (i,j),k in validate_ds_list[7].take(5):
        input_image = tf.expand_dims(i[6], 0)
        input_formula = tf.expand_dims(j[6], 0)
        context = model.return_context(input_image)
        formulas_encoded = model.Decoder.return_processed_text(input_formula)
        attention_output, scores = model.Decoder.return_after_attention(context, input_formula)
        print(attention_output.shape, scores)

#debugging()

if __name__ == '__main__':
    simple_training()


