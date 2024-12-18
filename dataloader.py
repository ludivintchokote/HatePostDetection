import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json
import random
import os
# import augly.text as txtaugs
# import augly.image as imaugs
#
# import nlpaug.augmenter.char as nac
# import nlpaug.augmenter.word as naw
# import nlpaug.augmenter.sentence as nas
# import nlpaug.flow as nafc
#
# from nlpaug.util import Action

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from statistics import mode
import easyocr

class meme_dataset(Dataset):
    def __init__(self, dataset_name, split, tokenizer, imga, texta, image_size: int=299, pad_length: int=100):
        print(dataset_name)
        self.split = split            
        self.img_attack = imga
        self.text_attack = texta
        self.global_path = '../../datasets/'
        print(self.split)
        split_file = os.path.join(self.global_path,dataset_name,'files_new/'+self.split+'.json')
        with open(split_file,'r') as f:
            self.data = json.load(f)
        self.img_path = os.path.join(self.global_path,dataset_name,'img/')
#         self.img_path = os.path.join(self.global_path,dataset_name,'newsprint')    
        #if not (self.img_attack=='original_text' or self.img_attack=='ocr'):
         #   self.img_path = os.path.join(self.global_path,dataset_name,self.img_attack)
        #else:
         #   self.img_path = os.path.join(self.global_path,dataset_name,'img')
        self.dataset = dataset_name
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.pad_length = pad_length
        self.transform = transforms.Compose([transforms.Resize((500,500)),transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
         
        self.NORMALIZE_LIST = ['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number']
        self.ANNOTATE_LIST = ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored']
        self.text_processor = TextPreProcessor(
            normalize= self.NORMALIZE_LIST,
            annotate= self.ANNOTATE_LIST,
            fix_html=True,
            segmenter="twitter", 
            unpack_hashtags=True,  
            unpack_contractions=True,  
            spell_correct_elong=True,  
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        )

        self.ocr = easyocr.Reader(['en']) 
    def __len__(self):
        return (len(self.data))
    
    
    def get_face_text(self,image_path):
        if image_path not in self.face_data:
            text_face = "no humans"
        else:
            text_face = ""
            for item in self.face_data[image_path]:
                sentence = str(item["race"]) + " " + str(item["gender"]) + " " + str(item["age"]) + " "
                text_face += sentence
        return text_face
    

    
    def get_web_text(self,image_name):
        web_path = os.path.join(self.web_path,image_name)
        web_path += '.json'
        with open (web_path,'rb') as f:
            web_file = json.load(f)
        list_web = []
        list_web.append(web_file['best_guess'][0])
        for entity in web_file['web_entities']:
            list_web.append(entity[1])

        text_web = " ".join(list_web)
        return text_web
    
    
    
    def fix_image(self,image):
        if len(np.array(image).shape) == 2:
            image = np.array(image)
            image = np.stack([image,image,image],axis=2)
            image = Image.fromarray(image)
        elif np.array(image).shape[2] == 4:
            image = np.array(image)
            image = image[:,:,:3]
            image = Image.fromarray(image)
        else:
            image = np.array(image)
            image = Image.fromarray(image)
        return image
    
    def get_image(self,path):
#         print(path)
        image = Image.open(path)
        image = self.fix_image(image)
#         if self.img_attack is not None:
#             image_np = np.array(image)
#             image = self.img_attack(image_np)
#         image = self.fix_image(image)
        image = self.transform(image)
        return image
    
    
    def get_labels(self,tweet_id):
        label = mode(self.gt[tweet_id]['labels'])
        if label>0:
            label = 1
        else:
            label = 0
        return label
    
    
#     def __getitem__(self, i):
#         image_path = os.path.join(self.img_path,self.data[i]['img'])
#         image = self.get_image(image_path)
#         text_string = 'text'
#         if not (self.text_attack=='original_text' or self.text_attack=='ocr' or self.text_attack==None):
#             if self.dataset == 'harmeme' and self.text_attack == 's&p_0.4':
#                 self.text_attack = 's&p0.4'
#             text_string += ('_'+self.text_attack)
#         elif self.text_attack=='ocr':
#             text_string += '_ocr'
#         #print(text_string)
#         #print(self.data[i])
#         tweet = self.data[i][text_string]
#         #print(tweet)
#         list_corrected_tweet = self.text_processor.pre_process_doc(tweet)
#         text_tweet = ' '.join(list_corrected_tweet)
#         text = text_tweet
#
# #         print(i,text)
# #         print(text)
# #         if self.split=='train':
# #             text_augs = [naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute"),
# #                          naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")]
# #             text_aug = random.choice(text_augs)
# #             text = text_aug.augment(text)
#         encoded = self.tokenizer.encode_plus(
#             text=text,  # the sentence to be encoded
#             add_special_tokens=True,  # Add [CLS] and [SEP]
#             max_length = 100,  # maximum length of a sentence
#             pad_to_max_length=True,
#             truncation=True,
#             return_tensors = 'pt',  # ask the function to return PyTorch tensors
#         )
#         single_label= self.data[i]['label']
# #         print(single_label)
#         sample = {'image':image, 'text':encoded, 'slabel':single_label, 'img_info': self.data[i]['img']}
#        # print(sample)
#         return sample

    def __getitem__(self, i):

        image_path = os.path.join(self.img_path, self.data[i]['img'])

        image = self.get_image(image_path)

        text_string = 'text'

        if not (self.text_attack=='original_text' or self.text_attack=='ocr' or self.text_attack==None):

            if self.dataset == 'harmeme' and self.text_attack == 's&p_0.4':

                self.text_attack = 's&p0.4'

            text_string += ('_'+self.text_attack)

        elif self.text_attack=='ocr':

            text_string += '_ocr'

        tweet = self.data[i][text_string]

        list_corrected_tweet = self.text_processor.pre_process_doc(tweet)

        text_tweet = ' '.join(list_corrected_tweet)

        text = text_tweet


        encoded = self.tokenizer.encode_plus(

            text=text,  # the sentence to be encoded

            add_special_tokens=True,  # Add [CLS] and [SEP]

            max_length = 100,  # maximum length of a sentence

            pad_to_max_length=True,

            truncation=True,

            return_tensors = 'pt',  # ask the function to return PyTorch tensors

        )

        # single_label= self.data[i]['label']

        sample = {'image':image, 'text':encoded, 'img_info': self.data[i]['img']}

        return sample