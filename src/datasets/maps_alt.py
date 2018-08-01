'''
Created on Sep 3, 2017

@author: ssudholt
'''
import os

import numpy as np
from skimage import io as img_io
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

import random
import scipy.io
from skimage.transform import resize

from cnn_ws.string_embeddings.phoc import build_phoc_descriptor, get_most_common_n_grams
from cnn_ws.transformations.image_size import check_size
from cnn_ws.transformations.homography_augmentation import HomographyAugmentation


class MAPSDataset(Dataset):
    '''
    PyTorch dataset class for the segmentation-based George Washington dataset
    '''

    def __init__(self, map_root_dir1, map_root_dir2, all_files,
                 embedding='phoc',
                 phoc_unigram_levels=(1, 2, 4, 8),
                 use_bigrams = False,
                 fixed_image_size=None,
                 min_image_width_height=30):
        '''
        Constructor

        :param gw_root_dir: full path to the GW root dir
        :param image_extension: the extension of image files (default: png)
        :param transform: which transform to use on the images
        :param cv_split_method: the CV method to be used for splitting the dataset
                                if None the entire dataset is used
        :param cv_split_idx: the index of the CV split to be used
        :param partition: the partition of the dataset (train or test)
                          can only be used if cv_split_method and cv_split_idx
                          is not None
        :param min_image_width_height: the minimum height or width a word image has to have
        '''
        # sanity checks
        if embedding not in ['phoc', 'spoc', 'dctow']:
            raise ValueError('embedding must be one of phoc, tsne, spoc or dctow')

        # class members
        self.word_list = None
        self.word_string_embeddings = None
        self.query_list = None
        self.label_encoder = None

        self.fixed_image_size = fixed_image_size

        #self.path = gw_root_dir

        #train_img_names = [line.strip() for line in open(os.path.join(gw_root_dir, 'old_sets/trainset.txt'))]
        #test_img_names = [line.strip() for line in open(os.path.join(gw_root_dir, 'old_sets/testset.txt'))]


        #train_test_mat = scipy.io.loadmat(os.path.join(gw_root_dir, 'IAM_words_indexes_sets.mat'))

        #gt_file = os.path.join(gw_root_dir, 'info.gtp')
        words = []
        train_split_ids = []
        test_split_ids = []
        cnt = 0

        for _file in all_files:
            A = np.load(map_root_dir1+'original_images_nopad_'+_file+'.tiff.npy')
            B = np.load(map_root_dir2+'original_words_nopad_'+_file+'.tiff.npy')

            for _id in range(len(A)):
                word_img = A[_id]
                transcr = B[_id]
                word_img = 1 - word_img.astype(np.float32) / 255.0
                word_img = check_size(img=word_img, min_image_width_height=min_image_width_height)
                word_img = np.transpose(word_img, (2, 0, 1))
                words.append((word_img, transcr.lower()))

        #self.train_ids = train_split_ids
        #self.test_ids = test_split_ids
        ratio = 0.7
        numTrain = int(len(words)*0.7)
        _ids_all = range(len(words))
        random.shuffle(_ids_all)
        train_ids = np.zeros(len(words))
        train_ids[_ids_all[0:numTrain]] = 1
        test_ids = 1-train_ids
        self.train_ids = train_ids
        self.test_ids = test_ids
        #self.train_ids = _ids_all[0:numTrain]
        #elf.test_ids = _ids_all[numTrain:]

        #self.train_ids = [x[0] for x in train_test_mat.get('idxTrain')]
        #self.test_ids = [x[0] for x in train_test_mat.get('idxTest')]

        self.words = words

        # compute a mapping from class string to class id
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([elem[1] for elem in words])



        # create embedding for the word_list
        self.word_embeddings = None
        word_strings = [elem[1] for elem in words]
        if embedding == 'phoc':
            # extract unigrams

            unigrams = [chr(i) for i in list(range(ord('a'), ord('z') + 1)) + list(range(ord('0'), ord('9') + 1))]
            # unigrams = get_unigrams_from_strings(word_strings=[elem[1] for elem in words])
            if use_bigrams:
                bigram_levels = [2]
                bigrams = get_most_common_n_grams(word_strings)
            else:
                bigram_levels = None
                bigrams = None

            self.word_embeddings = build_phoc_descriptor(words=word_strings,
                                                         phoc_unigrams=unigrams,
                                                         bigram_levels=bigram_levels,
                                                         phoc_bigrams=bigrams,
                                                         unigram_levels=phoc_unigram_levels)
        elif embedding == 'spoc':
            raise NotImplementedError()
        else:
            # dctow
            raise NotImplementedError()
        self.word_embeddings = self.word_embeddings.astype(np.float32)


    def mainLoader(self, partition=None, transforms=HomographyAugmentation()):

        self.transforms = transforms
        if partition not in [None, 'train', 'test']:
            raise ValueError('partition must be one of None, train or test')

        if partition is not None:
            if partition == 'train':
                #print(len(self.train_ids))
                self.word_list = [x for i, x in enumerate(self.words) if self.train_ids[i] == 1]
                #self.word_list = [x for i, x in enumerate(self.words) if i in train_ids]
                self.word_string_embeddings = [x for i, x in enumerate(self.word_embeddings) if self.train_ids[i] == 1]
                #self.word_string_embeddings = [x for i, x in enumerate(self.word_embeddings) if i in train_ids]
            else:
                self.word_list = [x for i, x in enumerate(self.words) if self.test_ids[i] == 1]
                self.word_string_embeddings = [x for i, x in enumerate(self.word_embeddings) if self.test_ids[i] == 1]
        else:
            # use the entire dataset
            self.word_list = self.words
            self.word_string_embeddings = self.word_embeddings

        if partition == 'test':
            # create queries
            word_strings = [elem[1] for elem in self.word_list]
            unique_word_strings, counts = np.unique(word_strings, return_counts=True)
            qry_word_ids = unique_word_strings[np.where(counts > 1)[0]]

            # remove stopwords if needed
            stopwords = []
            '''
            for line in open(os.path.join(self.path, 'iam-stopwords')):
                stopwords.append(line.strip().split(','))
            stopwords = stopwords[0]
            '''
            qry_word_ids = [word for word in qry_word_ids if word not in stopwords]

            query_list = np.zeros(len(word_strings), np.int8)
            qry_ids = [i for i in range(len(word_strings)) if word_strings[i] in qry_word_ids]
            query_list[qry_ids] = 1

            self.query_list = query_list
        else:
            word_strings = [elem[1] for elem in self.word_list]
            self.query_list = np.zeros(len(word_strings), np.int8)

        if partition == 'train':
            # weights for sampling
            #train_class_ids = [self.label_encoder.transform([self.word_list[index][1]]) for index in range(len(self.word_list))]
            #word_strings = [elem[1] for elem in self.word_list]
            unique_word_strings, counts = np.unique(word_strings, return_counts=True)
            ref_count_strings = {uword : count for uword, count in zip(unique_word_strings, counts)}
            weights = [1.0/ref_count_strings[word] for word in word_strings]
            self.weights = np.array(weights)/sum(weights)

            # neighbors
            #self.nbrs = NearestNeighbors(n_neighbors=32+1, algorithm='ball_tree').fit(self.word_string_embeddings)
            #indices = nbrs.kneighbors(self.word_embeddings, return_distance= False)




    def embedding_size(self):
        return len(self.word_string_embeddings[0])

    def __len__(self):
        return len(self.word_list)

    def __getitem__(self, index):
        word_img = self.word_list[index][0]
        if self.transforms is not None:
            word_img = self.transforms(word_img)

        # fixed size image !!!
        word_img = self._image_resize(word_img, self.fixed_image_size)

        word_img = word_img.reshape((1,) + word_img.shape)
        word_img = torch.from_numpy(word_img)
        embedding = self.word_string_embeddings[index]
        embedding = torch.from_numpy(embedding)
        class_id = self.label_encoder.transform([self.word_list[index][1]])
        is_query = self.query_list[index]

        return word_img, embedding, class_id, is_query

    # fixed sized image
    @staticmethod
    def _image_resize(word_img, fixed_img_size):

        if fixed_img_size is not None:
            if len(fixed_img_size) == 1:
                scale = float(fixed_img_size[0]) / float(word_img.shape[0])
                new_shape = (int(scale * word_img.shape[0]), int(scale * word_img.shape[1]))

            if len(fixed_img_size) == 2:
                new_shape = (fixed_img_size[0], fixed_img_size[1])

            word_img = resize(image=word_img, output_shape=new_shape).astype(np.float32)

        return word_img
