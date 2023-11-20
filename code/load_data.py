import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os

class load_train_data():
    def __init__(self, folder_path,input_date_data_size,device,do_mean=True,torch_manual_seed=100,np_random_seed=100):
        
        self.folder_path = folder_path
        self.input_date_data_size = input_date_data_size
        self.torch_manual_seed = torch_manual_seed
        self.np_random_seed = np_random_seed
        self.do_mean = do_mean
        self.device = device
        self.mean_x, self.std_x = 0, 0
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.feature_size = self.load_data()

    def load_data(self):
        load_df = pd.DataFrame()
        name_list=[]
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.folder_path, filename)
                df = pd.read_csv(file_path)
                df.replace("-", 1, inplace=True)
                df['即期點差'] = (df["即期賣出"] - df["即期買入"])*100 / df["即期買入"]
                df['現鈔點差'] = (df["現鈔賣出"] - df["現鈔買入"])*100 / df["現鈔買入"]
                df['點差差值'] = df['即期點差'] - df['現鈔點差']
                df['現買Return'] = (df["現鈔買入"] - df["現鈔買入"].shift(1)) / df["現鈔買入"].shift(1)
                df['現賣Return'] = (df["現鈔賣出"] - df["現鈔賣出"].shift(1)) / df["現鈔賣出"].shift(1)
                df['即買Return'] = (df["即期買入"] - df["即期買入"].shift(1)) / df["即期買入"].shift(1)
                df['即賣Return'] = (df["即期賣出"] - df["即期賣出"].shift(1)) / df["即期賣出"].shift(1)
                
                load_df = pd.concat([load_df, df["現鈔買入"], df["現鈔點差"], df["即期點差"], df['點差差值'], df['現買Return'], df['現賣Return'], df['即買Return'], df['即賣Return']], axis=1)
                filename = filename.replace(".csv", "")
                name_list.extend([f"{filename}現鈔買入", f"{filename}現鈔點差", f"{filename}即期點差", f"{filename}點差差值", f"{filename}現買Return", f"{filename}現賣Return", f"{filename}即買Return", f"{filename}即賣Return"])
        load_df.columns = name_list
        load_df.fillna(0, inplace=True)
        load_df = load_df.iloc[1:]
        load_df = load_df.iloc[::-1]
        load_df = load_df.reindex(sorted(load_df.columns), axis=1)
        return self.trun2torch(load_df)
    
    def trun2torch(self,df):
        torch.manual_seed(self.torch_manual_seed)
        np.random.seed(self.np_random_seed)
        train = df.to_numpy()
        train_size, feature_size = train.shape
        train_size = train_size - self.input_date_data_size


        train_x = np.empty([train_size, feature_size * self.input_date_data_size], dtype = float)
        train_y = np.empty([train_size, feature_size], dtype = float)

        for idx in range(train_size):
            temp_data = np.array([])
            for count in range(self.input_date_data_size):
                temp_data = np.hstack([temp_data, train[idx + count]])
            train_x[idx, :] = temp_data
            train_y[idx, :] = train[idx + self.input_date_data_size]

        train_y_columns = [df.columns.get_loc(col) for col in df.columns if '現鈔買入' in col]
        train_y = train_y[:,train_y_columns]
        
        if self.do_mean: train_x = self.mean(train_x)
        
        split_ratio = 0.2
        valid_size = int(train_size * split_ratio)

        indices = np.random.permutation(train_size)

        valid_indices = indices[:valid_size]
        train_indices = indices[valid_size:]

        valid_x = train_x[valid_indices]
        valid_y = train_y[valid_indices]

        train_x = train_x[train_indices]
        train_y = train_y[train_indices]

        train_x = torch.from_numpy(train_x.astype(np.float32)).to(self.device)
        train_y = torch.from_numpy(train_y.astype(np.float32)).to(self.device)
        valid_x = torch.from_numpy(valid_x.astype(np.float32)).to(self.device)
        valid_y = torch.from_numpy(valid_y.astype(np.float32)).to(self.device)

        return train_x, train_y, valid_x, valid_y, feature_size

    def mean(self,x):
        self.mean_x = np.mean(x, axis = 0)
        self.std_x = np.std(x, axis = 0)
        for i in range(len(x)):
            for j in range(len(x[0])):
                if self.std_x[j] != 0:
                    x[i][j] = (x[i][j] - self.mean_x[j]) / self.std_x[j]
        return x


       



class load_test_data():
    def __init__(self,folder_path,input_date_data_size,device,mean_x,std_x,do_mean=True):
        self.folder_path = folder_path
        self.input_date_data_size = input_date_data_size
        self.do_mean = do_mean
        self.device = device
        self.mean_x, self.std_x = mean_x, std_x
        self.test_x, self.feature_size = self.load_data()

    def load_data(self):
        load_df = pd.DataFrame()
        name_list=[]
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.folder_path, filename)
                df = pd.read_csv(file_path)
                df.replace("-", 1, inplace=True)
                diff_results = []
                for i in range(0, len(df), 4):
                    subset = df.iloc[i:i+4]
                    for j in range(4):
                        if j == 0:
                            diff_results.append([0, 0, 0, 0])
                        else:
                            diff = (subset.iloc[j][1:] - subset.iloc[j-1][1:]) / subset.iloc[j-1][1:]
                            diff_results.append(diff.tolist())

                diff_df_alternative = pd.DataFrame(diff_results, columns=['現買Return','現賣Return','即買Return','即賣Return'])
                df['現買Return'] = diff_df_alternative['現買Return'].values
                df['現賣Return'] = diff_df_alternative['現賣Return'].values
                df['即買Return'] = diff_df_alternative['即買Return'].values
                df['即賣Return'] = diff_df_alternative['即賣Return'].values
                df = df.drop(df.index[::4])
                df.reset_index(drop=True, inplace=True)
                df['即期點差'] = (df["即期賣出"] - df["即期買入"])*100 / df["即期買入"]
                df['現鈔點差'] = (df["現鈔賣出"] - df["現鈔買入"])*100 / df["現鈔買入"]
                df['點差差值'] = df['即期點差'] - df['現鈔點差']
                load_df = pd.concat([load_df, df["現鈔買入"], df["現鈔點差"], df["即期點差"], df['點差差值'], df['現買Return'], df['現賣Return'], df['即買Return'], df['即賣Return']], axis=1)
                filename = filename.replace(".csv", "")
                name_list.extend([f"{filename}現鈔買入", f"{filename}現鈔點差", f"{filename}即期點差", f"{filename}點差差值", f"{filename}現買Return", f"{filename}現賣Return", f"{filename}即買Return", f"{filename}即賣Return"])
        load_df.columns = name_list
        load_df.fillna(0, inplace=True)
        load_df = load_df.reindex(sorted(load_df.columns), axis=1)
        return self.trun2torch(load_df)

    def trun2torch(self,df):
        test = df.to_numpy()
        test_size, feature_size = test.shape
        test_size = test_size//self.input_date_data_size
        test_x = np.empty([test_size, feature_size * self.input_date_data_size], dtype = float)

        for idx in range(test_size):
            temp_data = np.array([])
            for count in range(self.input_date_data_size):
                temp_data = np.hstack([temp_data, test[idx * self.input_date_data_size + count]])
            test_x[idx, :] = temp_data
        if self.do_mean: test_x = self.mean(test_x)
        test_x = torch.from_numpy(test_x.astype(np.float32)).to(self.device) 
        return  test_x, feature_size

    def mean(self,x):
        for i in range(len(x)):
            for j in range(len(x[0])):
                if self.std_x[j] != 0:
                    x[i][j] = (x[i][j] - self.mean_x[j]) / self.std_x[j]
        return x
