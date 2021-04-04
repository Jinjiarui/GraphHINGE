import numpy as np
import random
import tqdm
import dgl
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
from itertools import *
import argparse

class MyDataset(Dataset):

    def __init__(self, user_pth, item_pth, user_metas, item_metas, x_data, y_data):

        self.user_pth = user_pth
        self.item_pth = item_pth
        self.UI = user_pth[user_metas[0]]
        self.IU = item_pth[item_metas[0]]
        self.UIUI = user_pth[user_metas[1]]
        self.IUIU = item_pth[item_metas[1]]
        self.UIAI1 = user_pth[user_metas[2]]
        self.IAIU1 = item_pth[item_metas[2]]
        self.UIAI2 = user_pth[user_metas[3]]
        self.IAIU2 = item_pth[item_metas[3]]
        self.UIAI3 = user_pth[user_metas[4]]
        self.IAIU3 = item_pth[item_metas[4]]
        self.x_data = x_data
        self.y_data = torch.Tensor(y_data).unsqueeze(1)
        self.len = x_data.shape[0]
    
    def __getitem__(self, index):
        return self.UI[self.x_data[index][0]]+1, self.IU[self.x_data[index][1]]+1, \
        self.UIUI[self.x_data[index][0]]+1, self.IUIU[self.x_data[index][1]]+1, \
        self.UIAI1[self.x_data[index][0]]+1, self.IAIU1[self.x_data[index][1]]+1, \
        self.UIAI2[self.x_data[index][0]]+1, self.IAIU2[self.x_data[index][1]]+1, \
        self.UIAI3[self.x_data[index][0]]+1, self.IAIU3[self.x_data[index][1]]+1, \
        self.y_data[index]

    def __len__(self):
        return self.len

class Dataloader:
    def __init__(self, data_path=None, name=None, pkl_path=None, saved=False, num_walks=20,walk_length=1, batch_size=128, is_topk = False, list_length=10, ratio=1):
        self._data = data_path
        self._name = name
        self._path = pkl_path
        self._num_walks_per_node = num_walks
        self._walk_length = walk_length
        self.saved = saved
        self.batch_size = batch_size
        self.is_topk = is_topk
        self.list_length = list_length
        self.ratio = ratio

    def generate_metapath(self,hg,head,meta_paths,path_names,path,name):
        dict={}
        for meta_path, path_name in zip(meta_paths, path_names):
            dict[path_name]={}
            for idx in tqdm.trange(hg.number_of_nodes(head)):
                traces, _ = dgl.sampling.random_walk(
                    hg, [idx] * self._num_walks_per_node, metapath=meta_path * self._walk_length)
                dict[path_name][idx]=traces.long()
        with open(os.path.join(path, name), 'wb') as file: 
            pkl.dump(dict, file)

    def split_data(self, hg, etype_name, user_item_src, user_item_dst,user_item_link, data_name):
        pos_label=[1]*user_item_link
        pos_data=list(zip(user_item_src,user_item_dst,pos_label))
        user_item_adj = np.array(hg.adj(etype=etype_name).to_dense())
        full_idx = np.where(user_item_adj==0)
        sample = random.sample(range(0, len(full_idx[0])), user_item_link)
        neg_label = [0]*user_item_link
        neg_data = list(zip(full_idx[0][sample],full_idx[1][sample],neg_label))
        full_data = pos_data + neg_data
        random.shuffle(full_data)

        train_size = int(len(full_data) * 0.6)
        eval_size = int(len(full_data) * 0.2)
        test_size = len(full_data) - train_size - eval_size
        train_data = full_data[:train_size]
        eval_data = full_data[train_size : train_size+eval_size]
        test_data = full_data[train_size+eval_size : train_size+eval_size+test_size]
        train_data = np.array(train_data)
        eval_data = np.array(eval_data)
        test_data = np.array(test_data)
        with open(os.path.join(self._path, data_name+'_train_1.pkl'), 'wb') as file: 
            pkl.dump(train_data, file)
        with open(os.path.join(self._path, data_name+'_test_1.pkl'), 'wb') as file: 
            pkl.dump(test_data, file)
        with open(os.path.join(self._path, data_name+'_eval_1.pkl'), 'wb') as file: 
            pkl.dump(eval_data, file)
        return train_data, eval_data, test_data
    
    def dataset_sample(self,data_name):
        train_file = open(self._path+'/'+data_name+'_train_1.pkl','rb')
        train_data = pkl.load(train_file)
        train_file.close()
        full_size=train_data.shape[0]
        sampled_idxs=random.sample(range(0,full_size), int(full_size*self.ratio))
        train_data=train_data[sampled_idxs,:]
        with open(os.path.join(self._path, data_name+'_train_'+str(self.ratio)+'.pkl'), 'wb') as file: 
            pkl.dump(train_data, file)

        eval_file = open(self._path+'/'+data_name+'_eval_1.pkl','rb')
        eval_data = pkl.load(eval_file)
        eval_file.close()
        full_size=eval_data.shape[0]
        sampled_idxs=random.sample(range(0,full_size), int(full_size*self.ratio))
        eval_data=eval_data[sampled_idxs,:]
        with open(os.path.join(self._path, data_name+'_eval_'+str(self.ratio)+'.pkl'), 'wb') as file: 
            pkl.dump(eval_data, file)
    
        test_file = open(self._path+'/'+data_name+'_test_1.pkl','rb')
        test_data = pkl.load(test_file)
        test_file.close()
        return train_data,eval_data,test_data

    def generate_topklist(self, data_name, test_data, list_length):
        users=test_data[:,0]
        user_idxs=np.unique(users)
        topk_list = []
        for user_idx in user_idxs:
            item_list = np.where(users==user_idx)[0].tolist()
            if len(item_list)>=list_length:
                sampled_items = random.sample(item_list, list_length)
                user_list=[user_idx]*list_length
                item_list=test_data[sampled_items,1]
                labels=test_data[sampled_items,2]
                topk_list = topk_list+ list(zip(user_list, item_list, labels))
        topk_list = np.array(topk_list)
        with open(os.path.join(self._path, data_name+'_topk_'+str(list_length)+'_.pkl'), 'wb') as file: 
            pkl.dump(topk_list, file)
        return topk_list              
    
    def load_data(self):
        if self._name == 'amazon':
            return self._load_amazon()
        elif self._name == 'movielens':
            return self._load_movielens()
        elif self._name == 'yelp':
            return self._load_yelp()
        elif self._name == 'aminer':
            return self._load_aminer()
        elif self._name == 'dblp':
            return self._load_dblp()
        elif self._name == 'douban_book':
            return self._load_douban_book()
        elif self._name == 'amazon_book':
            return self._load_amazon_book()
        elif self._name == 'movielens_20m':
            return self._load_movielens_20m()
        else:
            raise NotImplementedError

    def _load_amazon(self):
        # User-Item 3584 2753 50903 UIUI
        # Item-View 2753 3857 5694 UIVI
        # Item-Brand 2753 334 2753 UIBI
        # Item-Category 2753 22 5508 UICI
        if self._data == None:
            self._data = '../data/Amazon'

        #Load or construct Graph
        if (os.path.exists(os.path.join(self._path, 'amazon_hg.pkl'))):
            hg_file = open(os.path.join(self._path, 'amazon_hg.pkl'),'rb')
            hg = pkl.load(hg_file)
            hg_file.close()
            print("Graph Loaded.")
        else:
            #Construct graph from raw data.
            # load data of amazon
            _data_list = ['item_view.dat', 'user_item.dat', 'item_brand.dat', 'item_category.dat']

            # item_view
            item_view_src=[]
            item_view_dst=[]
            with open(self._data + '/' + _data_list[0]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split(',')
                    _item, _view= int(_line[0]), int(_line[1])
                    item_view_src.append(_item)
                    item_view_dst.append(_view)

            # user_item
            user_item_src=[]
            user_item_dst=[]
            user_item_link=0
            with open(self._data + '/' + _data_list[1]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _user, _item, _rate = int(_line[0]), int(_line[1]), int(_line[2])
                    if _rate > 3:
                        user_item_src.append(_user)
                        user_item_dst.append(_item)  
                        user_item_link = user_item_link + 1

            # item_brand
            item_brand_src=[]
            item_brand_dst=[]
            with open(self._data + '/' + _data_list[2]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split(',')
                    _item, _brand= int(_line[0]), int(_line[1])
                    item_brand_src.append(_item)
                    item_brand_dst.append(_brand)
        
            # item_category
            item_category_src=[]
            item_category_dst=[]
            with open(self._data + '/' + _data_list[3]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split(',')
                    _item, _category= int(_line[0]), int(_line[1])
                    item_category_src.append(_item)
                    item_category_dst.append(_category)

            #build graph
            hg = dgl.heterograph({
                ('item', 'iv', 'view') : (item_view_src, item_view_dst),
                ('view', 'vi', 'item') : (item_view_dst, item_view_src),
                ('user', 'ui', 'item') : (user_item_src, user_item_dst),
                ('item', 'iu', 'user') : (user_item_dst, user_item_src),
                ('item', 'ib', 'brand') : (item_brand_src, item_brand_dst), 
                ('brand', 'bi', 'item') : (item_brand_dst, item_brand_src),
                ('item', 'ic', 'category') : (item_category_src, item_category_dst),
                ('category', 'ci', 'item') : (item_category_dst, item_category_src)})

            with open(os.path.join(self._path, 'amazon_hg.pkl'), 'wb') as file: 
                pkl.dump(hg, file)
            print("Graph constructed.")
        
        #Split dataset
        etype_name = 'ui' #predict edge type
        if (os.path.exists(os.path.join(self._path, 'amazon_train_'+str(self.ratio)+'.pkl'))):
            train_file = open(self._path+'/'+'amazon_train_'+str(self.ratio)+'.pkl','rb')
            train_data = pkl.load(train_file)
            train_file.close()
            test_file = open(self._path+'/'+'amazon_test_1.pkl','rb')
            test_data = pkl.load(test_file)
            test_file.close()
            eval_file = open(self._path+'/'+'amazon_eval_'+str(self.ratio)+'.pkl','rb')
            eval_data = pkl.load(eval_file)
            eval_file.close()
            print("Train, eval, and test loaded.")
        else:
            if self.ratio == 1:
                train_data, eval_data, test_data = self.split_data(hg, etype_name, user_item_src, user_item_dst,user_item_link,'amazon')
            else:
                if (os.path.exists(os.path.join(self._path, 'amazon_train_1.pkl'))) == False:
                    train_data, eval_data, test_data = self.split_data(hg, etype_name, user_item_src, user_item_dst,user_item_link,'amazon')
                train_data, eval_data, test_data = self.dataset_sample('amazon')
            print("Train, eval, and test splited.")

        #Prepare dataset
        #Define meta-paths.
        scale='_'+str(self._num_walks_per_node)+'_'+str(self._walk_length) 
        user_paths=[['ui'],['ui', 'iu', 'ui'], ['ui', 'iv', 'vi'], ['ui', 'ib', 'bi'], ['ui', 'ic', 'ci']]
        item_paths=[['iu'],['iu', 'ui', 'iu'], ['iv', 'vi', 'iu'], ['ib', 'bi', 'iu'], ['ic', 'ci', 'iu']]
        user_metas=['UI','UIUI', 'UIVI', 'UIBI', 'UICI']
        item_metas=['IU','IUIU', 'IVIU', 'IBIU', 'ICIU']
        user_pkl='amazon_user'+scale+'.pkl'
        item_pkl='amazon_item'+scale+'.pkl'

        #Generate paths.
        if self.saved == True:
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'item',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")

        if not (os.path.exists(self._path+'/'+user_pkl)):
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'item',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")

        print("Load paths from:")
        print(user_pkl)
        print(item_pkl)
        user_file = open(self._path+'/'+user_pkl,'rb')
        user_pth = pkl.load(user_file)
        user_file.close()
        item_file = open(self._path+'/'+item_pkl,'rb')
        item_pth = pkl.load(item_file)
        item_file.close()
        train_set = MyDataset(user_pth, item_pth, user_metas, item_metas, train_data[:,:2], train_data[:,2])
        eval_set = MyDataset(user_pth, item_pth, user_metas, item_metas, eval_data[:,:2], eval_data[:,2])
        test_set = MyDataset(user_pth, item_pth, user_metas, item_metas, test_data[:,:2], test_data[:,2])
        train_loader= DataLoader(dataset=train_set, batch_size = self.batch_size, shuffle=True)
        eval_loader= DataLoader(dataset=eval_set, batch_size = self.batch_size, shuffle=True)
        test_loader= DataLoader(dataset=test_set, batch_size = self.batch_size, shuffle=True)

        #Prepare topk test set.
        if self.is_topk == True:
            if (os.path.exists(os.path.join(self._path, 'amazon_topk_'+str(self.list_length)+'_.pkl'))):
                topk_file = open(self._path+'/'+'amazon_topk_'+str(self.list_length)+'_.pkl','rb')
                topk_list = pkl.load(topk_file)
                topk_file.close()
                print("Top K loaded.")
            else:
                topk_list = self.generate_topklist('amazon',test_data, list_length = self.list_length)
                print("Top K generated.")
            topk_set = MyDataset(user_pth, item_pth, user_metas, item_metas, topk_list[:,:2], topk_list[:,2])
            test_loader= DataLoader(dataset=topk_set, batch_size = self.batch_size, shuffle=False)
        return user_metas, item_metas, train_loader, eval_loader, test_loader, hg.num_nodes('user'), hg.num_nodes('item'), hg.num_nodes('view'), hg.num_nodes('brand'), hg.num_nodes('category')
        
    def _load_movielens(self):
        # User-Movie 943 1682 100000 UMUM
        # User-Age 943 8 943 UAUM
        # User-Occupation 943 21 943 UOUM
        # Movie-Genre 1682 18 2861 UMGM
        # _data_list = os.listdir(self._data)
        if self._data == None:
            self._data = '../data/Movielens'

        #Load or construct Graph
        if (os.path.exists(os.path.join(self._path, 'movielens_hg.pkl'))):
            hg_file = open(os.path.join(self._path, 'movielens_hg.pkl'),'rb')
            hg = pkl.load(hg_file)
            hg_file.close()
            print("Graph Loaded.")
        else:
            #Construct graph from raw data.
            # load data of movielens
            _data_list = ['movie_genre.dat', 'user_occupation.dat', 'user_age.dat', 'user_movie.dat']

            # movie_genre
            movie_genre_src=[]
            movie_genre_dst=[]
            with open(self._data + '/' + _data_list[0]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _movie, _genre = int(_line[0]), int(_line[1])
                    movie_genre_src.append(_movie)
                    movie_genre_dst.append(_genre)

            # user_movie
            user_movie_src=[]
            user_movie_dst=[]
            user_movie_link=0
            with open(self._data + '/' + _data_list[3]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _user, _item, _rate = int(_line[0]), int(_line[1]), int(_line[2])
                    if _rate > 3:
                        user_movie_src.append(_user)
                        user_movie_dst.append(_item)  
                        user_movie_link = user_movie_link + 1

            # user_occupation
            user_occupation_src=[]
            user_occupation_dst=[]
            with open(self._data + '/' + _data_list[1]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _user, _occupation = int(_line[0]), int(_line[1])
                    user_occupation_src.append(_user)
                    user_occupation_dst.append(_occupation)

            # user_age
            user_age_src=[]
            user_age_dst=[]
            with open(self._data + '/' + _data_list[2]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _user, _age = int(_line[0]), int(_line[1])
                    user_age_src.append(_user)
                    user_age_dst.append(_age)

            #build graph
            hg = dgl.heterograph({
                ('movie', 'mg', 'genre') : (movie_genre_src, movie_genre_dst),
                ('genre', 'gm', 'movie') : (movie_genre_dst, movie_genre_src),
                ('user', 'um', 'movie') : (user_movie_src, user_movie_dst),
                ('movie', 'mu', 'user') : (user_movie_dst, user_movie_src),
                ('user', 'uo', 'occupation') : (user_occupation_src, user_occupation_dst), 
                ('occupation', 'ou', 'user') : (user_occupation_dst, user_occupation_src),
                ('user', 'ua', 'age') : (user_age_src, user_age_dst),
                ('age', 'au', 'user') : (user_age_dst, user_age_src)})

            with open(os.path.join(self._path, 'movielens_hg.pkl'), 'wb') as file: 
                pkl.dump(hg, file)
            print("Graph constructed.")

        # Split data.
        z=hg.edges(etype= ('user', 'um', 'movie'))
        etype_name = 'um'
        user_item_src = z[0].numpy().tolist()
        user_item_dst = z[1].numpy().tolist()
        user_item_link = hg.num_edges('um')
        if (os.path.exists(os.path.join(self._path, 'movielens_train_'+str(self.ratio)+'.pkl'))):
            train_file = open(self._path+'/'+'movielens_train_'+str(self.ratio)+'.pkl','rb')
            train_data = pkl.load(train_file)
            train_file.close()
            test_file = open(self._path+'/'+'movielens_test_1.pkl','rb')
            test_data = pkl.load(test_file)
            test_file.close()
            eval_file = open(self._path+'/'+'movielens_eval_'+str(self.ratio)+'.pkl','rb')
            eval_data = pkl.load(eval_file)
            eval_file.close()
            print("Train, eval, and test loaded.")
        else:
            if self.ratio == 1:
                train_data, eval_data, test_data = self.split_data(hg, etype_name, user_item_src, user_item_dst,user_item_link,'movielens')
            else:
                if (os.path.exists(os.path.join(self._path, 'movielens_train_1.pkl'))) == False:
                    train_data, eval_data, test_data = self.split_data(hg, etype_name, user_item_src, user_item_dst,user_item_link,'movielens')
                train_data, eval_data, test_data = self.dataset_sample('movielens')
            print("Train, eval, and test splited.")

        #Prepare dataset
        #Define meta-paths.
        scale='_'+str(self._num_walks_per_node)+'_'+str(self._walk_length)
        user_paths=[['um'],['um', 'mu', 'um'], ['um', 'mg', 'gm'], ['um', 'mg', 'gm'], ['um', 'mg', 'gm']]
        item_paths=[['mu'],['mu', 'um', 'mu'], ['mg', 'gm', 'mu'], ['mg', 'gm', 'mu'], ['mg', 'gm', 'mu']]
        user_metas=['UI','UIUI', 'UIVI', 'UIBI', 'UICI']
        item_metas=['IU','IUIU', 'IVIU', 'IBIU', 'ICIU']
        user_pkl='movielens_user'+scale+'.pkl'
        item_pkl='movielens_item'+scale+'.pkl'

        #Generate paths.
        if self.saved == True:
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'movie',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")

        if not (os.path.exists(self._path+'/'+user_pkl)):
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'movie',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")
        
        print("Load paths from:")
        print(user_pkl)
        print(item_pkl)
        user_file = open(self._path+'/'+user_pkl,'rb')
        user_pth = pkl.load(user_file)
        user_file.close()
        item_file = open(self._path+'/'+item_pkl,'rb')
        item_pth = pkl.load(item_file)
        item_file.close()
        train_set = MyDataset(user_pth, item_pth, user_metas, item_metas, train_data[:,:2], train_data[:,2])
        eval_set = MyDataset(user_pth, item_pth, user_metas, item_metas, eval_data[:,:2], eval_data[:,2])
        test_set = MyDataset(user_pth, item_pth, user_metas, item_metas, test_data[:,:2], test_data[:,2])
        train_loader= DataLoader(dataset=train_set, batch_size = self.batch_size, shuffle=True)
        eval_loader= DataLoader(dataset=eval_set, batch_size = self.batch_size, shuffle=True)
        test_loader= DataLoader(dataset=test_set, batch_size = self.batch_size, shuffle=True)

        #Prepare topk test set.
        if self.is_topk == True:
            if (os.path.exists(os.path.join(self._path, 'movielens_topk_'+str(self.list_length)+'_.pkl'))):
                topk_file = open(self._path+'/'+'movielens_topk_'+str(self.list_length)+'_.pkl','rb')
                topk_list = pkl.load(topk_file)
                topk_file.close()
                print("Top K loaded.")
            else:
                topk_list = self.generate_topklist('movielens',test_data, list_length = self.list_length)
                print("Top K generated.")
            topk_set = MyDataset(user_pth, item_pth, user_metas, item_metas, topk_list[:,:2], topk_list[:,2])
            test_loader= DataLoader(dataset=topk_set, batch_size = self.batch_size, shuffle=False)
        return user_metas, item_metas, train_loader, eval_loader, test_loader, hg.num_nodes('user'), hg.num_nodes('movie'), hg.num_nodes('genre'), hg.num_nodes('genre'), hg.num_nodes('genre')

    def _load_yelp(self):
        #business: 14284             user: 16239 
        #category(genre): 511         compliment: 11 
        #city: 47             
        #business-genre: 40009     business-city: 14267 
        #user-business: 198397         user-city: 76875 
        #user-user: 158590

        if self._data == None:
            self._data = '../data/Yelp'

        #Load or construct graph
        if (os.path.exists(os.path.join(self._path, 'yelp_hg.pkl'))):
            hg_file = open(os.path.join(self._path, 'yelp_hg.pkl'),'rb')
            hg = pkl.load(hg_file)
            hg_file.close()
            print("Graph Loaded.")
        else:
            #Construct graph from raw data.
            # load data of yelp
            _data_list = ['business_category.dat', 'business_city.dat', 'user_business.dat', 'user_compliment.dat', 'user_user.dat']

            #business_category
            business_category_src=[]
            business_category_dst=[]
            with open(self._data + '/' + _data_list[0]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _business, _category= int(_line[0]), int(_line[1])
                    business_category_src.append(_business)
                    business_category_dst.append(_category)

            #business_city
            business_city_src=[]
            business_city_dst=[]
            with open(self._data + '/' + _data_list[1]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _business, _city= int(_line[0]), int(_line[1])
                    business_city_src.append(_business)
                    business_city_dst.append(_city)

            #user_business
            user_business_src=[]
            user_business_dst=[]
            user_item_link=0
            with open(self._data + '/' + _data_list[2]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _user, _business= int(_line[0]), int(_line[1])
                    user_business_src.append(_user)
                    user_business_dst.append(_business)
                    user_item_link+=1

            #user_compliment
            user_compliment_src=[]
            user_compliment_dst=[]
            with open(self._data + '/' + _data_list[3]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _user, _compliment= int(_line[0]), int(_line[1])
                    user_compliment_src.append(_user)
                    user_compliment_dst.append(_compliment)

            #user_user
            user_user_src=[]
            user_user_dst=[]
            with open(self._data + '/' + _data_list[4]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _user1, _user2= int(_line[0]), int(_line[1])
                    user_user_src.append(_user1)
                    user_user_dst.append(_user2)

            #build graph
            hg = dgl.heterograph({
                ('business', 'bc', 'category') : (business_category_src, business_category_dst),
                ('category', 'cb', 'business') : (business_category_dst, business_category_src),
                ('business', 'bci', 'city') : (business_city_src, business_city_dst),
                ('city', 'cib', 'business') : (business_city_dst, business_city_src),
                ('user', 'ub', 'business') : (user_business_src, user_business_dst),
                ('business', 'bu', 'user') : (user_business_dst, user_business_src),
                ('user', 'uc', 'compliment') : (user_compliment_src, user_compliment_dst),
                ('compliment', 'cu', 'user') : (user_compliment_dst, user_compliment_src),
                ('user', 'uu', 'user') : (user_user_src+user_user_dst, user_user_dst+user_user_src)
            })

            with open(os.path.join(self._path, 'yelp_hg.pkl'), 'wb') as file: 
                pkl.dump(hg, file)
            print("Graph constructed.")

        #Split dataset
        etype_name = 'ub' #predict edge type
        if (os.path.exists(os.path.join(self._path, 'yelp_train_'+str(self.ratio)+'.pkl'))):
            train_file = open(self._path+'/'+'yelp_train_'+str(self.ratio)+'.pkl','rb')
            train_data = pkl.load(train_file)
            train_file.close()
            test_file = open(self._path+'/'+'yelp_test_1.pkl','rb')
            test_data = pkl.load(test_file)
            test_file.close()
            eval_file = open(self._path+'/'+'yelp_eval_'+str(self.ratio)+'.pkl','rb')
            eval_data = pkl.load(eval_file)
            eval_file.close()
            print("Train, eval, and test loaded.")
        else:
            if self.ratio == 1:
                train_data, eval_data, test_data = self.split_data(hg, etype_name, user_business_src, user_business_dst,user_item_link,'yelp')
            else:
                if (os.path.exists(os.path.join(self._path, 'yelp_train_1.pkl'))) == False:
                    train_data, eval_data, test_data = self.split_data(hg, etype_name, user_business_src, user_business_dst,user_item_link,'yelp')
                train_data, eval_data, test_data = self.dataset_sample('yelp')
            print("Train, eval, and test splited.")
            
        #Prepare dataset
        #Define meta-paths.
        scale='_'+str(self._num_walks_per_node)+'_'+str(self._walk_length) 
        user_paths=[['ub'],['ub', 'bu', 'ub'], ['ub', 'bc', 'cb'], ['ub', 'bci', 'cib'], ['ub', 'bc', 'cb']]
        item_paths=[['bu'],['bu', 'ub', 'bu'], ['bc', 'cb', 'bu'], ['bci', 'cib', 'bu'], ['bc', 'cb', 'bu']]
        user_metas=['UI','UIUI', 'UIVI', 'UIBI', 'UICI']
        item_metas=['IU','IUIU', 'IVIU', 'IBIU', 'ICIU']
        user_pkl='yelp_user'+scale+'.pkl'
        item_pkl='yelp_item'+scale+'.pkl'

        #Generate paths.
        if self.saved == True:
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'business',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")

        if not (os.path.exists(self._path+'/'+user_pkl)):
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'business',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")

        print("Load paths from:")
        print(user_pkl)
        print(item_pkl)
        user_file = open(self._path+'/'+user_pkl,'rb')
        user_pth = pkl.load(user_file)
        user_file.close()
        item_file = open(self._path+'/'+item_pkl,'rb')
        item_pth = pkl.load(item_file)
        item_file.close()
        train_set = MyDataset(user_pth, item_pth, user_metas, item_metas, train_data[:,:2], train_data[:,2])
        eval_set = MyDataset(user_pth, item_pth, user_metas, item_metas, eval_data[:,:2], eval_data[:,2])
        test_set = MyDataset(user_pth, item_pth, user_metas, item_metas, test_data[:,:2], test_data[:,2])
        train_loader= DataLoader(dataset=train_set, batch_size = self.batch_size, shuffle=True)
        eval_loader= DataLoader(dataset=eval_set, batch_size = self.batch_size, shuffle=True)
        test_loader= DataLoader(dataset=test_set, batch_size = self.batch_size, shuffle=True)

        #Prepare topk test set.
        if self.is_topk == True:
            if (os.path.exists(os.path.join(self._path, 'yelp_topk_'+str(self.list_length)+'_.pkl'))):
                topk_file = open(self._path+'/'+'yelp_topk_'+str(self.list_length)+'_.pkl','rb')
                topk_list = pkl.load(topk_file)
                topk_file.close()
                print("Top K loaded.")
            else:
                topk_list = self.generate_topklist('yelp',test_data, list_length = self.list_length)
                print("Top K generated.")
            topk_set = MyDataset(user_pth, item_pth, user_metas, item_metas, topk_list[:,:2], topk_list[:,2])
            test_loader= DataLoader(dataset=topk_set, batch_size = self.batch_size, shuffle=False)
        return user_metas, item_metas, train_loader, eval_loader, test_loader, hg.num_nodes('user'), hg.num_nodes('business'), hg.num_nodes('category'), hg.num_nodes('city'), hg.num_nodes('category')
        

    def _load_dblp(self):

        if self._data == None:
            self._data = '../data/DBLP'

        #Load or construct graph
        if (os.path.exists(os.path.join(self._path, 'dblp_hg.pkl'))):
            hg_file = open(os.path.join(self._path, 'dblp_hg.pkl'),'rb')
            hg = pkl.load(hg_file)
            hg_file.close()
            print("Graph Loaded.")
        else:
            #Construct graph from raw data.
            # load data of dblp
            _data_list = ['author_label.dat', 'paper_author.dat', 'paper_conference.dat', 'paper_type.dat']

            #author_label
            author_label_src=[]
            author_label_dst=[]
            with open(self._data + '/' + _data_list[0]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _author, _label= int(_line[0]), int(_line[1])
                    author_label_src.append(_author)
                    author_label_dst.append(_label)

            #paper_author
            paper_author_src=[]
            paper_author_dst=[]
            user_item_link=0
            with open(self._data + '/' + _data_list[1]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _paper, _author= int(_line[0]), int(_line[1])
                    paper_author_src.append(_paper)
                    paper_author_dst.append(_author)
                    user_item_link+=1

            #paper_conference
            paper_conference_src=[]
            paper_conference_dst=[]
            with open(self._data + '/' + _data_list[2]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _paper, _conference= int(_line[0]), int(_line[1])
                    paper_conference_src.append(_paper)
                    paper_conference_dst.append(_conference)

            #paper_type
            paper_type_src=[]
            paper_type_dst=[]
            with open(self._data + '/' + _data_list[3]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _paper, _type= int(_line[0]), int(_line[1])
                    paper_type_src.append(_paper)
                    paper_type_dst.append(_type)


            #build graph
            hg = dgl.heterograph({
                ('author', 'al', 'label') : (author_label_src, author_label_dst),
                ('label', 'la', 'author') : (author_label_dst, author_label_src),
                ('paper', 'pa', 'author') : (paper_author_src, paper_author_dst),
                ('author', 'ap', 'paper') : (paper_author_dst, paper_author_src),
                ('paper', 'pc', 'conference') : (paper_conference_src, paper_conference_dst),
                ('conference', 'cp', 'paper') : (paper_conference_dst, paper_conference_src),
                ('paper', 'pt', 'type') : (paper_type_src, paper_type_dst),
                ('type', 'tp', 'paper') : (paper_type_dst, paper_type_src)
            })

            with open(os.path.join(self._path, 'dblp_hg.pkl'), 'wb') as file: 
                pkl.dump(hg, file)
            print("Graph constructed.")

        #Split dataset
        etype_name = 'ap' #predict edge type
        if (os.path.exists(os.path.join(self._path, 'dblp_train_'+str(self.ratio)+'.pkl'))):
            train_file = open(self._path+'/'+'dblp_train_'+str(self.ratio)+'.pkl','rb')
            train_data = pkl.load(train_file)
            train_file.close()
            test_file = open(self._path+'/'+'dblp_test_1.pkl','rb')
            test_data = pkl.load(test_file)
            test_file.close()
            eval_file = open(self._path+'/'+'dblp_eval_'+str(self.ratio)+'.pkl','rb')
            eval_data = pkl.load(eval_file)
            eval_file.close()
            print("Train, eval, and test loaded.")
        else:
            if self.ratio == 1:
                train_data, eval_data, test_data = self.split_data(hg, etype_name, paper_author_dst, paper_author_src,user_item_link,'dblp')
            else:
                if (os.path.exists(os.path.join(self._path, 'dblp_train_1.pkl'))) == False:
                    train_data, eval_data, test_data = self.split_data(hg, etype_name, paper_author_dst, paper_author_src,user_item_link,'dblp')
                train_data, eval_data, test_data = self.dataset_sample('dblp')
            print("Train, eval, and test splited.")
            
        #Prepare dataset
        #Define meta-paths.
        scale='_'+str(self._num_walks_per_node)+'_'+str(self._walk_length) 
        user_paths=[['ap'],['ap', 'pa', 'ap'], ['ap', 'pc', 'cp'], ['ap', 'pt', 'tp'], ['ap', 'pc', 'cp']]
        item_paths=[['pa'],['pa', 'ap', 'pa'], ['pc', 'cp', 'pa'], ['pt', 'tp', 'pa'], ['pc', 'cp', 'pa']]
        user_metas=['UI','UIUI', 'UIVI', 'UIBI', 'UICI']
        item_metas=['IU','IUIU', 'IVIU', 'IBIU', 'ICIU']
        user_pkl='dblp_user'+scale+'.pkl'
        item_pkl='dblp_item'+scale+'.pkl'

        #Generate paths.
        if self.saved == True:
            self.generate_metapath(hg,'author',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'paper',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")

        if not (os.path.exists(self._path+'/'+user_pkl)):
            self.generate_metapath(hg,'author',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'paper',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")

        print("Load paths from:")
        print(user_pkl)
        print(item_pkl)
        user_file = open(self._path+'/'+user_pkl,'rb')
        user_pth = pkl.load(user_file)
        user_file.close()
        item_file = open(self._path+'/'+item_pkl,'rb')
        item_pth = pkl.load(item_file)
        item_file.close()
        train_set = MyDataset(user_pth, item_pth, user_metas, item_metas, train_data[:,:2], train_data[:,2])
        eval_set = MyDataset(user_pth, item_pth, user_metas, item_metas, eval_data[:,:2], eval_data[:,2])
        test_set = MyDataset(user_pth, item_pth, user_metas, item_metas, test_data[:,:2], test_data[:,2])
        train_loader= DataLoader(dataset=train_set, batch_size = self.batch_size, shuffle=True)
        eval_loader= DataLoader(dataset=eval_set, batch_size = self.batch_size, shuffle=True)
        test_loader= DataLoader(dataset=test_set, batch_size = self.batch_size, shuffle=True)

        #Prepare topk test set.
        if self.is_topk == True:
            if (os.path.exists(os.path.join(self._path, 'dblp_topk_'+str(self.list_length)+'_.pkl'))):
                topk_file = open(self._path+'/'+'dblp_topk_'+str(self.list_length)+'_.pkl','rb')
                topk_list = pkl.load(topk_file)
                topk_file.close()
                print("Top K loaded.")
            else:
                topk_list = self.generate_topklist('dblp',test_data, list_length = self.list_length)
                print("Top K generated.")
            topk_set = MyDataset(user_pth, item_pth, user_metas, item_metas, topk_list[:,:2], topk_list[:,2])
            test_loader= DataLoader(dataset=topk_set, batch_size = self.batch_size, shuffle=False)
        return user_metas, item_metas, train_loader, eval_loader, test_loader, hg.num_nodes('author'), hg.num_nodes('paper'), hg.num_nodes('conference'), hg.num_nodes('type'), hg.num_nodes('conference')

    def _load_douban_book(self):
        # User-Book 13024 22347 792062
        # User-Group 13024 2936 1189271
        # User-User 13024 13024 169150
        # User-Location 13024 38 10592
        # Book-Author 22347 10805 21907
        # Book-Publisher 22347 1815 21773
        # Book-Year 22347 64 2192
        # _data_list = os.listdir(self._data)
        if self._data == None:
            self._data = '../data/Douban Book'

        #Load or construct Graph
        if (os.path.exists(os.path.join(self._path, 'douban_book_hg.pkl'))):
            hg_file = open(os.path.join(self._path, 'douban_book_hg.pkl'),'rb')
            hg = pkl.load(hg_file)
            hg_file.close()
            print("Graph Loaded.")
        else:
            # load data of douban book
            _data_list = ['book_year.dat', 'book_author.dat', 'user_book.dat', 'book_publisher.dat', 'user_group.dat', 'user_location.dat', 'user_user.dat']

            # user_book
            user_book_src=[]
            user_book_dst=[]
            user_book_link=0
            with open(self._data + '/' + _data_list[2]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _user, _book, _rate = int(_line[0]), int(_line[1]), int(_line[2])
                    if _rate > 3:
                        user_book_src.append(_user)
                        user_book_dst.append(_book)
                        user_book_link=user_book_link+1

            # book_author
            book_author_src=[]
            book_author_dst=[]
            with open(self._data + '/' + _data_list[1]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _book, _author = int(_line[0]), int(_line[1])
                    book_author_src.append(_book)
                    book_author_dst.append(_author)

            # book_publisher
            book_publisher_src=[]
            book_publisher_dst=[]
            with open(self._data + '/' + _data_list[3]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _book, _publisher = int(_line[0]), int(_line[1])
                    book_publisher_src.append(_book)
                    book_publisher_dst.append(_publisher)

            #book_year
            book_year_src=[]
            book_year_dst=[]
            with open(self._data + '/' + _data_list[0]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _book, _year = int(_line[0]), int(_line[1])
                    book_year_src.append(_book)
                    book_year_dst.append(_year)

            # user_group
            user_group_src=[]
            user_group_dst=[]
            with open(self._data + '/' + _data_list[4]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _user, _group = int(_line[0]), int(_line[1])
                    user_group_src.append(_user)
                    user_group_dst.append(_group)

            # user_user
            user_user_src=[]
            user_user_dst=[]
            with open(self._data + '/' + _data_list[6]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split('\t')
                    _user1, _user2 = int(_line[0]), int(_line[1])
                    user_user_src.append(_user1)
                    user_user_dst.append(_user2)

            #build graph
            hg = dgl.heterograph({
                ('user', 'ub', 'book') : (user_book_src, user_book_dst),
                ('book', 'bu', 'user') : (user_book_dst, user_book_src),
                ('book', 'ba', 'author') : (book_author_src, book_author_dst),
                ('author', 'ab', 'book') : (book_author_dst, book_author_src),
                ('book', 'bp', 'publisher') : (book_publisher_src, book_publisher_dst), 
                ('publisher', 'pb', 'book') : (book_publisher_dst, book_publisher_src),
                ('book', 'by', 'year') : (book_year_src, book_year_dst),
                ('year', 'yb', 'book') : (book_year_dst, book_year_src)})

            with open(os.path.join(self._path, 'douban_book_hg.pkl'), 'wb') as file: 
                pkl.dump(hg, file)
            print("Graph constructed.")

        # Split data.
        z=hg.edges(etype= ('user', 'ub', 'book'))
        etype_name = 'ub'
        user_item_src = z[0].numpy().tolist()
        user_item_dst = z[1].numpy().tolist()
        user_item_link = hg.num_edges('ub')
        if (os.path.exists(os.path.join(self._path, 'douban_book_train_'+str(self.ratio)+'.pkl'))):
            train_file = open(self._path+'/'+'douban_book_train_'+str(self.ratio)+'.pkl','rb')
            train_data = pkl.load(train_file)
            train_file.close()
            test_file = open(self._path+'/'+'douban_book_test_1.pkl','rb')
            test_data = pkl.load(test_file)
            test_file.close()
            eval_file = open(self._path+'/'+'douban_book_eval_'+str(self.ratio)+'.pkl','rb')
            eval_data = pkl.load(eval_file)
            eval_file.close()
            print("Train, eval, and test loaded.")
        else:
            if self.ratio == 1:
                train_data, eval_data, test_data = self.split_data(hg, etype_name, user_item_src, user_item_dst,user_item_link, 'douban_book')
            else:
                if (os.path.exists(os.path.join(self._path, 'douban_book_train_1.pkl'))) == False:
                    train_data, eval_data, test_data = self.split_data(hg, etype_name, user_item_src, user_item_dst,user_item_link, 'douban_book')
                train_data, eval_data, test_data = self.dataset_sample('douban_book')
            print("Train, eval, and test splited.")

        #Prepare dataset
        #Define meta-paths.
        #sample path
        scale='_'+str(self._num_walks_per_node)+'_'+str(self._walk_length)
        user_paths=[['ub'],['ub', 'bu', 'ub'], ['ub', 'ba', 'ab'], ['ub', 'bp', 'pb'], ['ub', 'by', 'yb']]
        item_paths=[['bu'],['bu', 'ub', 'bu'], ['ba', 'ab', 'bu'], ['bp', 'pb', 'bu'], ['by', 'yb', 'bu']]
        user_metas=['UB','UBUB', 'UBAB', 'UBPB', 'UBYB']
        item_metas=['BU','BUBU', 'BABU', 'BPBU', 'BYBU']
        user_pkl='douban_book_user'+scale+'.pkl'
        item_pkl='douban_book_item'+scale+'.pkl'

        #Generate paths.
        if self.saved == True:
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'book',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")

        if not (os.path.exists(self._path+'/'+user_pkl)):
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'book',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")
        
        print("Load paths from:")
        print(user_pkl)
        print(item_pkl)
        user_file = open(self._path+'/'+user_pkl,'rb')
        user_pth = pkl.load(user_file)
        user_file.close()
        item_file = open(self._path+'/'+item_pkl,'rb')
        item_pth = pkl.load(item_file)
        item_file.close()
        train_set = MyDataset(user_pth, item_pth, user_metas, item_metas, train_data[:,:2], train_data[:,2])
        eval_set = MyDataset(user_pth, item_pth, user_metas, item_metas, eval_data[:,:2], eval_data[:,2])
        test_set = MyDataset(user_pth, item_pth, user_metas, item_metas, test_data[:,:2], test_data[:,2])
        train_loader= DataLoader(dataset=train_set, batch_size = self.batch_size, shuffle=True)
        eval_loader= DataLoader(dataset=eval_set, batch_size = self.batch_size, shuffle=True)
        test_loader= DataLoader(dataset=test_set, batch_size = self.batch_size, shuffle=True)

        #Prepare topk test set.
        if self.is_topk == True:
            if (os.path.exists(os.path.join(self._path, 'douban_book_topk_'+str(self.list_length)+'_.pkl'))):
                topk_file = open(self._path+'/'+'douban_book_topk_'+str(self.list_length)+'_.pkl','rb')
                topk_list = pkl.load(topk_file)
                topk_file.close()
                print("Top K loaded.")
            else:
                topk_list = self.generate_topklist('douban_book',test_data, list_length = self.list_length)
                print("Top K generated.")
            topk_set = MyDataset(user_pth, item_pth, user_metas, item_metas, topk_list[:,:2], topk_list[:,2])
            test_loader= DataLoader(dataset=topk_set, batch_size = self.batch_size, shuffle=False)
        return user_metas, item_metas, train_loader, eval_loader, test_loader, hg.num_nodes('user'), hg.num_nodes('book'), hg.num_nodes('author'), hg.num_nodes('publisher'), hg.num_nodes('year')

    def _load_amazon_book(self):
        if self._data == None:
            self._data = '../data/pickled_data'

        if (os.path.exists(os.path.join(self._path, 'amazon_book_sampled.pkl'))):
            hg_file = open(os.path.join(self._path, 'amazon_book_sampled.pkl'),'rb')
            hg = pkl.load(hg_file)
            hg_file.close()
            print("Sampled Graph loaded!")
            print("User: %d" %(hg.num_nodes('user')))
            print("Item: %d" %(hg.num_nodes('item')))
            print("Knowledge: %d" %(hg.num_nodes('knowledge')))
            print("User-Item: %d" %(hg.num_edges('ui')))
            print("Item-Knowledge: %d" %(hg.num_edges('ik')))
        else:
            # dict_keys(['train_data', 'eval_data', 'test_data', 'adj', 'adj_relational', 'kg_dict'])
            _data_amazon_book = pkl.load(open(self._data + '/' + 'ab.pkl', 'rb'))
            train_data = _data_amazon_book['train_data']
            test_data = _data_amazon_book['test_data']
            eval_data = _data_amazon_book['eval_data']
            kg_dict = _data_amazon_book['kg_dict']

            full_data = np.concatenate((train_data,test_data,eval_data), 0)

            #user_item
            pos_data = full_data[full_data[:,2] == 1]
            user_item_src = pos_data[:,0].tolist()
            user_item_dst = pos_data[:,1].tolist()

            #item_knowledge
            item_knowledge_src = []
            item_knowledge_dst = []
            know_num = 0
            item_num1 = max(kg_dict.keys())
            for kg_item, kg_list in zip(kg_dict.keys(), kg_dict.values()):
                for kg_knowledge in kg_list:
                    item_knowledge_src.append(kg_item)
                    item_knowledge_dst.append(kg_knowledge)
                    know_num =max(know_num,kg_knowledge)

            #build graph
            hg = dgl.heterograph({
                ('user', 'ui', 'item') : (user_item_src, user_item_dst),
                ('item', 'iu', 'user') : (user_item_dst, user_item_src),
                ('item', 'ik', 'knowledge') : (item_knowledge_src, item_knowledge_dst),
                ('knowledge', 'ki', 'item') : (item_knowledge_dst, item_knowledge_src)})
            user_sample = random.sample(range(0, hg.num_nodes('user')), hg.num_nodes('user')-20000)
            hg = dgl.remove_nodes(hg, user_sample, ntype='user')
            adj_ui = np.array(hg.adj(etype='ui').to_dense())
            aux = adj_ui.sum(0)
            item_removed = np.where(aux==0)
            item_removed = torch.Tensor(item_removed).squeeze(1).squeeze(0).long()
            hg = dgl.remove_nodes(hg, item_removed, ntype='item')
            adj_ik = np.array(hg.adj(etype='ik').to_dense())
            aux = adj_ik.sum(0)
            know_removed = np.where(aux==0)
            know_removed = torch.Tensor(know_removed).squeeze(1).squeeze(0).long()
            hg = dgl.remove_nodes(hg, know_removed, ntype='knowledge')
            adj_ui = np.array(hg.adj(etype='ui').to_dense())
            aux = adj_ui.sum(1)
            user_removed = np.where(aux==0)
            user_removed = torch.Tensor(user_removed).squeeze(1).squeeze(0).long()
            hg = dgl.remove_nodes(hg, user_removed, ntype='user')

            with open(os.path.join(self._path, 'amazon_book_sampled.pkl'), 'wb') as file: 
                pkl.dump(hg, file)
            print("Sample Graph Finished.")
            print("User: %d" %(hg.num_nodes('user')))
            print("Item: %d" %(hg.num_nodes('item')))
            print("Knowledge: %d" %(hg.num_nodes('knowledge')))
            print("User-Item: %d" %(hg.num_edges('ui')))
            print("Item-Knowledge: %d" %(hg.num_edges('ik')))

        # Split data.
        z=hg.edges(etype= ('user', 'ui', 'item'))
        etype_name = 'ui'
        user_item_src = z[0].numpy().tolist()
        user_item_dst = z[1].numpy().tolist()
        user_item_link = hg.num_edges('ui')
        if (os.path.exists(os.path.join(self._path, 'amazon_book_train_'+str(self.ratio)+'.pkl'))):
            train_file = open(self._path+'/'+'amazon_book_train_'+str(self.ratio)+'.pkl','rb')
            train_data = pkl.load(train_file)
            train_file.close()
            test_file = open(self._path+'/'+'amazon_book_test_1.pkl','rb')
            test_data = pkl.load(test_file)
            test_file.close()
            eval_file = open(self._path+'/'+'amazon_book_eval_'+str(self.ratio)+'.pkl','rb')
            eval_data = pkl.load(eval_file)
            eval_file.close()
            print("Train, eval, and test loaded.")
        else:
            if self.ratio == 1:
                train_data, eval_data, test_data = self.split_data(hg, etype_name, user_item_src, user_item_dst,user_item_link, 'amazon_book')
            else:
                if (os.path.exists(os.path.join(self._path, 'amazon_book_train_1.pkl'))) == False:
                    train_data, eval_data, test_data = self.split_data(hg, etype_name, user_item_src, user_item_dst,user_item_link, 'amazon_book')
                train_data, eval_data, test_data = self.dataset_sample('amazon_book')
            print("Train, eval, and test splited.")

        #Prepare dataset
        #Define meta-paths.
        #sample path
        scale='_'+str(self._num_walks_per_node)+'_'+str(self._walk_length) 
        user_paths=[['ui'],['ui', 'iu', 'ui'], ['ui', 'ik', 'ki'], ['ui', 'ik', 'ki'], ['ui', 'ik', 'ki']]
        item_paths=[['iu'],['iu', 'ui', 'iu'], ['ik', 'ki', 'iu'], ['ik', 'ki', 'iu'], ['ik', 'ki', 'iu']]
        user_metas=['UI','UIUI', 'UIVI', 'UIBI', 'UICI']
        item_metas=['IU','IUIU', 'IVIU', 'IBIU', 'ICIU']
        user_pkl='amazon_book_user'+scale+'.pkl'
        item_pkl='amazon_book_item'+scale+'.pkl'

        #Generate paths.
        if self.saved == True:
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'item',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")

        if not (os.path.exists(self._path+'/'+user_pkl)):
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'item',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")
        
        print("Load paths from:")
        print(user_pkl)
        print(item_pkl)
        user_file = open(self._path+'/'+user_pkl,'rb')
        user_pth = pkl.load(user_file)
        user_file.close()
        item_file = open(self._path+'/'+item_pkl,'rb')
        item_pth = pkl.load(item_file)
        item_file.close()
        train_set = MyDataset(user_pth, item_pth, user_metas, item_metas, train_data[:,:2], train_data[:,2])
        eval_set = MyDataset(user_pth, item_pth, user_metas, item_metas, eval_data[:,:2], eval_data[:,2])
        test_set = MyDataset(user_pth, item_pth, user_metas, item_metas, test_data[:,:2], test_data[:,2])
        train_loader= DataLoader(dataset=train_set, batch_size = self.batch_size, shuffle=True)
        eval_loader= DataLoader(dataset=eval_set, batch_size = self.batch_size, shuffle=True)
        test_loader= DataLoader(dataset=test_set, batch_size = self.batch_size, shuffle=True)

        #Prepare topk test set.
        if self.is_topk == True:
            if (os.path.exists(os.path.join(self._path, 'amazon_book_topk_'+str(self.list_length)+'_.pkl'))):
                topk_file = open(self._path+'/'+'amazon_book_topk_'+str(self.list_length)+'_.pkl','rb')
                topk_list = pkl.load(topk_file)
                topk_file.close()
                print("Top K loaded.")
            else:
                topk_list = self.generate_topklist('amazon_book',test_data, list_length = self.list_length)
                print("Top K generated.")
            topk_set = MyDataset(user_pth, item_pth, user_metas, item_metas, topk_list[:,:2], topk_list[:,2])
            test_loader= DataLoader(dataset=topk_set, batch_size = self.batch_size, shuffle=False)

        return user_metas, item_metas, train_loader, eval_loader, test_loader, hg.num_nodes('user'), hg.num_nodes('item'), hg.num_nodes('knowledge'), hg.num_nodes('knowledge'), hg.num_nodes('knowledge')

    def _load_movielens_20m(self):
        if self._data == None:
            self._data = '../data/pickled_data'

        if (os.path.exists(os.path.join(self._path, 'movielens_20m_sampled.pkl'))):
            hg_file = open(os.path.join(self._path, 'movielens_20m_sampled.pkl'),'rb')
            hg = pkl.load(hg_file)
            hg_file.close()
            print("Sampled Graph loaded!")
            print("User: %d" %(hg.num_nodes('user')))
            print("Item: %d" %(hg.num_nodes('item')))
            print("Knowledge: %d" %(hg.num_nodes('knowledge')))
            print("User-Item: %d" %(hg.num_edges('ui')))
            print("Item-Knowledge: %d" %(hg.num_edges('ik')))
        else:
            # dict_keys(['train_data', 'eval_data', 'test_data', 'adj', 'adj_relational', 'kg_dict'])
            _data_movielens_20m = pkl.load(open(self._data + '/' + 'ml-20m.pkl', 'rb'))
            train_data = _data_movielens_20m['train_data']
            test_data = _data_movielens_20m['test_data']
            eval_data = _data_movielens_20m['eval_data']
            kg_dict = _data_movielens_20m['kg_dict']

            full_data = np.concatenate((train_data,test_data,eval_data), 0)

            #user_item
            pos_data = full_data[full_data[:,2] == 1]
            user_item_src = pos_data[:,0].tolist()
            user_item_dst = pos_data[:,1].tolist()

            #item_knowledge
            item_knowledge_src = []
            item_knowledge_dst = []
            know_num = 0
            item_num1 = max(kg_dict.keys())
            for kg_item, kg_list in zip(kg_dict.keys(), kg_dict.values()):
                for kg_knowledge in kg_list:
                    item_knowledge_src.append(kg_item)
                    item_knowledge_dst.append(kg_knowledge)
                    know_num =max(know_num,kg_knowledge)

            #build graph
            hg = dgl.heterograph({
                ('user', 'ui', 'item') : (user_item_src, user_item_dst),
                ('item', 'iu', 'user') : (user_item_dst, user_item_src),
                ('item', 'ik', 'knowledge') : (item_knowledge_src, item_knowledge_dst),
                ('knowledge', 'ki', 'item') : (item_knowledge_dst, item_knowledge_src)})
            user_sample = random.sample(range(0, hg.num_nodes('user')), hg.num_nodes('user')-20000)
            hg = dgl.remove_nodes(hg, user_sample, ntype='user')
            adj_ui = np.array(hg.adj(etype='ui').to_dense())
            aux = adj_ui.sum(0)
            item_removed = np.where(aux==0)
            item_removed = torch.Tensor(item_removed).squeeze(1).squeeze(0).long()
            hg = dgl.remove_nodes(hg, item_removed, ntype='item')
            adj_ik = np.array(hg.adj(etype='ik').to_dense())
            aux = adj_ik.sum(0)
            know_removed = np.where(aux==0)
            know_removed = torch.Tensor(know_removed).squeeze(1).squeeze(0).long()
            hg = dgl.remove_nodes(hg, know_removed, ntype='knowledge')
            adj_ui = np.array(hg.adj(etype='ui').to_dense())
            aux = adj_ui.sum(1)
            user_removed = np.where(aux==0)
            user_removed = torch.Tensor(user_removed).squeeze(1).squeeze(0).long()
            hg = dgl.remove_nodes(hg, user_removed, ntype='user')

            with open(os.path.join(self._path, 'movielens_20m_sampled.pkl'), 'wb') as file: 
                pkl.dump(hg, file)
            print("Sample Graph Finished.")
            print("User: %d" %(hg.num_nodes('user')))
            print("Item: %d" %(hg.num_nodes('item')))
            print("Knowledge: %d" %(hg.num_nodes('knowledge')))
            print("User-Item: %d" %(hg.num_edges('ui')))
            print("Item-Knowledge: %d" %(hg.num_edges('ik')))

        # Split data.
        z=hg.edges(etype= ('user', 'ui', 'item'))
        etype_name = 'ui'
        user_item_src = z[0].numpy().tolist()
        user_item_dst = z[1].numpy().tolist()
        user_item_link = hg.num_edges('ui')
        if (os.path.exists(os.path.join(self._path, 'movielens_20m_train_'+str(self.ratio)+'.pkl'))):
            train_file = open(self._path+'/'+'movielens_20m_train_'+str(self.ratio)+'.pkl','rb')
            train_data = pkl.load(train_file)
            train_file.close()
            test_file = open(self._path+'/'+'movielens_20m_test_1.pkl','rb')
            test_data = pkl.load(test_file)
            test_file.close()
            eval_file = open(self._path+'/'+'movielens_20m_eval_'+str(self.ratio)+'.pkl','rb')
            eval_data = pkl.load(eval_file)
            eval_file.close()
            print("Train, eval, and test loaded.")
        else:
            if self.ratio == 1:
                train_data, eval_data, test_data = self.split_data(hg, etype_name, user_item_src, user_item_dst,user_item_link, 'movielens_20m')
            else:
                if (os.path.exists(os.path.join(self._path, 'movielens_20m_train_1.pkl'))) == False:
                    train_data, eval_data, test_data = self.split_data(hg, etype_name, user_item_src, user_item_dst,user_item_link, 'movielens_20m')
                train_data, eval_data, test_data = self.dataset_sample('movielens_20m')
            print("Train, eval, and test splited.")

        #Prepare dataset
        #Define meta-paths.
        #sample path
        scale='_'+str(self._num_walks_per_node)+'_'+str(self._walk_length) 
        user_paths=[['ui'],['ui', 'iu', 'ui'], ['ui', 'ik', 'ki'], ['ui', 'ik', 'ki'], ['ui', 'ik', 'ki']]
        item_paths=[['iu'],['iu', 'ui', 'iu'], ['ik', 'ki', 'iu'], ['ik', 'ki', 'iu'], ['ik', 'ki', 'iu']]
        user_metas=['UI','UIUI', 'UIVI', 'UIBI', 'UICI']
        item_metas=['IU','IUIU', 'IVIU', 'IBIU', 'ICIU']
        user_pkl='movielens_20m_user'+scale+'.pkl'
        item_pkl='movielens_20m_item'+scale+'.pkl'

        #Generate paths.
        if self.saved == True:
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'item',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")

        if not (os.path.exists(self._path+'/'+user_pkl)):
            self.generate_metapath(hg,'user',user_paths, user_metas, self._path, user_pkl)
            self.generate_metapath(hg,'item',item_paths, item_metas, self._path, item_pkl)
            print("Paths sampled.")
        
        print("Load paths from:")
        print(user_pkl)
        print(item_pkl)
        user_file = open(self._path+'/'+user_pkl,'rb')
        user_pth = pkl.load(user_file)
        user_file.close()
        item_file = open(self._path+'/'+item_pkl,'rb')
        item_pth = pkl.load(item_file)
        item_file.close()
        train_set = MyDataset(user_pth, item_pth, user_metas, item_metas, train_data[:,:2], train_data[:,2])
        eval_set = MyDataset(user_pth, item_pth, user_metas, item_metas, eval_data[:,:2], eval_data[:,2])
        test_set = MyDataset(user_pth, item_pth, user_metas, item_metas, test_data[:,:2], test_data[:,2])
        train_loader= DataLoader(dataset=train_set, batch_size = self.batch_size, shuffle=True)
        eval_loader= DataLoader(dataset=eval_set, batch_size = self.batch_size, shuffle=True)
        test_loader= DataLoader(dataset=test_set, batch_size = self.batch_size, shuffle=True)

        #Prepare topk test set.
        if self.is_topk == True:
            if (os.path.exists(os.path.join(self._path, 'movielens_20m_topk_'+str(self.list_length)+'_.pkl'))):
                topk_file = open(self._path+'/'+'movielens_20m_topk_'+str(self.list_length)+'_.pkl','rb')
                topk_list = pkl.load(topk_file)
                topk_file.close()
                print("Top K loaded.")
            else:
                topk_list = self.generate_topklist('movielens_20m',test_data, list_length = self.list_length)
                print("Top K generated.")
            topk_set = MyDataset(user_pth, item_pth, user_metas, item_metas, topk_list[:,:2], topk_list[:,2])
            test_loader= DataLoader(dataset=topk_set, batch_size = self.batch_size, shuffle=False)

        return user_metas, item_metas, train_loader, eval_loader, test_loader, hg.num_nodes('user'), hg.num_nodes('item'), hg.num_nodes('knowledge'), hg.num_nodes('knowledge'), hg.num_nodes('knowledge')

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset')

    parser.add_argument('-d', type=str,help='Dataset.')
    parser.add_argument('-p', type=str,help='Dataset path.')
    parser.add_argument('-o', type=str,default='../data/data_out',help='Output path.')
    parser.add_argument('-b', type=int, default=128, help='Batch size.')
    parser.add_argument('-s', type=int,default=0, help='Sample path to be saved.')
    parser.add_argument('-n', type=int, default=20,help='Num of walks per node.')
    parser.add_argument('-w', type=int, default=1,help='Scale of walk length.')
    parser.add_argument('-ratio', type=float, default=1,help='Sample ratio.')
    args = parser.parse_args()

    Dataloader(args.p, args.d, args.o, args.s, args.n, args.w, args.b,ratio=args.ratio).load_data()
 