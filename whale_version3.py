from collections import defaultdict
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPool2D, Input, Dense, Flatten, Dropout, GlobalMaxPooling2D, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.layers import merge
from PIL import Image
from keras.models import Model
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import os
import glob


model_name = "triplet_model"
file_path = model_name + "weights.best.hdf5"


# =================== Generate triplet tuple ===================
class SampleGenerator(object):
    def __init__(self, file_label_mapping, other_class='new_whale'):
        self.file_label_mapping = file_label_mapping
        self.label_file_dict = defaultdict(list)
        self.other_class = []

        # key: filename, value: label
        self.list_all_files = list(file_label_mapping.keys())
        self.range_all_files = list(range(len(self.list_all_files)))

        for file, class_ in file_label_mapping.items():
            if class_ == other_class:
                self.other_class.append(file)
            else:
                self.label_file_dict[class_].append(file)

        self.list_classes = list(set(self.file_label_mapping.values()))
        self.range_list_classes = range(len(self.list_classes))

        # use sample number as weight
        self.class_weight = np.array([len(self.label_file_dict[class_]) for class_ in self.list_classes])

        # normalization
        self.class_weight = self.class_weight / np.sum(self.class_weight)


    def get_sample(self):
        # random get 2 positive example and 1 negative example
        class_idx = np.random.choice(self.range_list_classes, 1, p=self.class_weight)[0]

        # pick a class and then pick 2 image
        examples_class_idx = np.random.choice(range(len(self.label_file_dict[self.list_classes[class_idx]])), 2)

        positive_example_1 = self.label_file_dict[self.list_classes[class_idx]][examples_class_idx[0]]
        positive_example_2 = self.label_file_dict[self.list_classes[class_idx]][examples_class_idx[1]]

        # pick a image with different class
        negative_example = None
        while negative_example is None or self.file_label_mapping[negative_example] == self.file_label_mapping[positive_example_1]:
            negative_example_idx = np.random.choice(self.range_all_files, 1)[0]
            negative_example = self.list_all_files[negative_example_idx]
        return positive_example_1, negative_example, positive_example_2


data = pd.read_csv('train.csv')
train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=1337)

train_filename_label_dict = {k: v for k, v in zip(train.Image.values, train.Id.values)}
test_filename_label_dict = {k: v for k, v in zip(test.Image.values, test.Id.values)}

train_generator = SampleGenerator(train_filename_label_dict)
test_generator = SampleGenerator(test_filename_label_dict)

# print(train_generator.get_sample())
# print(test_generator.get_sample())


# =================== Build model ===================
class BuildModel():
    def __init__(self):
        self.batch_size = 8
        self.input_shape = (256, 256)
        self.base_path = 'train/'

    def identity_loss(self, y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)

    def triplet_loss(self, X):
        positive_item_latent, negative_item_latent, user_latent = X

        # loss
        loss = 1. - K.sigmoid(
            K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
            K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True)
        )
        return loss

    def gen(self, triplet_gen):
        while True:
            list_positive_examples_1 = []
            list_negative_examples = []
            list_positive_examples_2 = []

            for i in range(self.batch_size):
                positive_example_1, negative_example, positive_example_2 = triplet_gen.get_sample()

                positive_example_1_img = self.read_and_resize(self.base_path + positive_example_1)
                negative_example_img = self.read_and_resize(self.base_path + negative_example)
                positive_example_2_img = self.read_and_resize(self.base_path + positive_example_2)

                positive_example_1_img = self.augment(positive_example_1_img)
                negative_example_img = self.augment(negative_example_img)
                positive_example_2_img = self.augment(positive_example_2_img)

                list_positive_examples_1.append(positive_example_1_img)
                list_negative_examples.append(negative_example_img)
                list_positive_examples_2.append(positive_example_2_img)

            list_positive_examples_1 = np.array(list_positive_examples_1)
            list_negative_examples = np.array(list_negative_examples)
            list_positive_examples_2 = np.array(list_positive_examples_2)
            yield [list_positive_examples_1, list_negative_examples, list_positive_examples_2], np.ones(self.batch_size)

    # Use ResNet as base model
    def get_base_model(self):
        latent_dim = 50
        base_model = ResNet50(include_top=False, weights='imagenet')
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        dense_1 = Dense(latent_dim)(x)
        normalized = Lambda(lambda x: K.l2_normalize(x, axis=1))(dense_1)
        dense_2 = Dense(latent_dim)(normalized)
        normalized_2 = Lambda(lambda x: K.l2_normalize(x, axis=1))(dense_2)
        base_model = Model(base_model.input, normalized_2, name='Base')
        return base_model


    def read_and_resize(self, filepath):
        img = Image.open((filepath)).convert('RGB')
        img = img.resize(self.input_shape)
        img_array = np.array(img, dtype='uint8')[..., ::-1]
        return np.array(img_array/ (np.max(img_array) + 0.001), dtype='float32')

    def augment(self, img_array):
        if np.random.uniform(0, 1) > 0.9:
            img_array = np.fliplr(img_array)
        return img_array


    def data_generator(self, fpaths, batch=16):
        i = 0
        imgs, fnames = [], []
        for path in fpaths:
            if i == 0:
                imgs, fnames = [], []
            i += 1
            img = self.read_and_resize(path)
            imgs.append(img)
            fnames.append(os.path.basename(path))
            if i == batch:
                i = 0
                imgs = np.array(imgs)
                yield fnames, imgs
        if i < batch:
            imgs = np.array(imgs)
            yield fnames, imgs
        raise StopIteration()


if __name__ == '__main__':
    bm = BuildModel()

# ====================== train ======================
    base_model = bm.get_base_model()

    positive_example_1 = Input(bm.input_shape + (3,), name='positive_1')
    negative_example = Input(bm.input_shape + (3,), name='negative')
    positive_example_2 = Input(bm.input_shape + (3,), name='positive_2')

    positive_example_1_out = base_model(positive_example_1)
    negative_example_out = base_model(negative_example)
    positive_example_2_out = base_model(positive_example_2)
    loss = merge(
        [positive_example_1_out, negative_example_out, positive_example_2_out],
        mode=bm.triplet_loss,
        name='loss',
        output_shape=(1,)
    )

    model = Model(
        inputs=[positive_example_1, negative_example, positive_example_2],
        outputs=loss
    )
    model.compile(loss=bm.identity_loss, optimizer=Adam(0.0001))
    print(model.summary())

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    callbacks_list = [checkpoint, early]

    if (not os.path.exists(file_path)):
        model.fit_generator(
            bm.gen(train_generator),
            validation_data=bm.gen(test_generator),
            epochs=8,
            verbose=2,
            workers=4,
            use_multiprocessing=True,
            callbacks=callbacks_list,
            steps_per_epoch=300,
            validation_steps=30
        )
    else:
        model.load_weights(file_path)

# ====================== predict ======================

    file_id_mapping = {k: v for k, v in zip(data.Image.values, data.Id.values)}
    train_files = glob.glob('train/*.jpg')
    test_files = glob.glob('test/*.jpg')

    train_preds = []
    train_file_names = []

    i = 1
    for fnames, imgs in bm.data_generator(train_files, batch=32):
        i += 1
        predicts = base_model.predict(imgs)
        predicts = predicts.tolist()
        train_preds += predicts
        train_file_names += fnames
        print('Train data computing complete: ', round(i * 32 / len(train_files) * 100, 0), '%')


    train_preds = np.array(train_preds)

    test_preds = []
    test_file_names = []
    i = 1
    for fnames, imgs in bm.data_generator(test_files, batch=32):
        i += 1
        predicts = base_model.predict(imgs)
        predicts = predicts.tolist()
        test_preds += predicts
        test_file_names += fnames
        print('Test data computing complete: ', round(i * 32 / len(test_files) * 100, 0), '%')

    test_preds = np.array(test_preds)

    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(train_preds)
    distances, neighbors = neigh.kneighbors(train_preds)

    print('Distance:', distances)
    print('Neighbors', neighbors)

    distances_test, neighbors_test = neigh.kneighbors(test_preds)

    distances_test, neighbors_test = distances_test.tolist(), neighbors_test.tolist()

    preds_str = []

    for filepath, distance, neighbour_ in zip(test_file_names, distances_test, neighbors_test):
        sample_result = []
        sample_classes = []
        for d, n in zip(distance, neighbour_):
            train_file = train_files[n].split(os.sep)[-1]
            class_train = file_id_mapping[train_file]
            sample_classes.append(class_train)
            sample_result.append((class_train, d))

        if "new_whale" not in sample_classes:
            sample_result.append(("new_whale", 0.1))
        sample_result.sort(key=lambda x: x[1])
        sample_result = sample_result[:5]
        preds_str.append(" ".join([x[0] for x in sample_result]))

    df = pd.DataFrame(preds_str, columns=["Id"])
    df['Image'] = [x.split(os.sep)[-1] for x in test_file_names]
    df.to_csv("sub.csv", index=False)


