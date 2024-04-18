import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features


class DatasetEttHour(Dataset):
    """
     A dataset class for handling hourly data from the ETT dataset for time series forecasting and imputation tasks.

     This dataset supports loading and preprocessing hourly data for training, testing, and validation purposes. It provides
     functionality to scale the data, encode time features, and specify the percentage of data to use.

     Parameters:
         root_path (str): Root directory where the dataset is stored.
         flag (str): Specifies the dataset split to load. Valid options are 'train', 'test', or 'validation'.
         size (List[int], optional): Specifies the sequence length, label length, and prediction length. Defaults to [24*4*4, 24*4, 24*4].
         features (str, optional): Indicates the type of features to use ('S' for single feature or 'M' for multiple features). Defaults to 'S'.
         data_path (str, optional): Path to the specific dataset file. Defaults to 'ETTh1.csv'.
         target (str, optional): Target variable name. This parameter is not relevant for imputation tasks. Defaults to 'OT'.
         scale (bool, optional): If True, scales the data using StandardScaler. Defaults to True.
         timeenc (int, optional): Encoding for time features (0 for no encoding, 1 for encoding). Defaults to 0.
         freq (str, optional): Frequency of the time series data (e.g., 'h' for hourly). Defaults to 'h'.
         percent (int, optional): Percentage of the dataset to use, mainly for testing purposes. Defaults to 10.

     The dataset is expected to be in CSV format with a column named 'date' for timestamps and additional columns for features.
     """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=10):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'validation']
        type_map = {'train': 0, 'validation': 1, 'test': 2}
        self.set_type = type_map[flag]

        # self.features = features
        # self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        df_data = None
        # if self.features == 'M' or self.features == 'MS':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        # elif self.features == 'S':
        #     df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(columns=['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            data_stamp = None

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class DatasetEttMinute(Dataset):
    """
    A dataset class for handling minute-level data from the ETT dataset for time series forecasting and imputation tasks.

    This dataset supports loading and preprocessing minute-level data for training, testing, and validation purposes. It provides
    functionality to scale the data, encode time features, and specify the percentage of data to use.

    Parameters:
        root_path (str): Root directory where the dataset is stored.
        flag (str): Specifies the dataset split to load. Valid options are 'train', 'test', or 'validation'.
        size (List[int], optional): Specifies the sequence length, label length, and prediction length. Defaults to [24*4*4, 24*4, 24*4].
        data_path (str, optional): Path to the specific dataset file. Defaults to 'ETTm1.csv'.
        scale (bool, optional): If True, scales the data using StandardScaler. Defaults to True.
        timeenc (int, optional): Encoding for time features (0 for no encoding, 1 for encoding). Defaults to 0.
        freq (str, optional): Frequency of the time series data (e.g., 't' for minute-level data). Defaults to 't'.
        seasonal_patterns (List[int], optional): Seasonal patterns for time series data. Defaults to None.
        percent (int, optional): Percentage of the dataset to use, mainly for testing purposes. Defaults to 10.

    The dataset is expected to be in CSV format with a column named 'date' for timestamps and additional columns for features.
    """
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTm1.csv',
                 scale=True, timeenc=0, freq='t',
                 seasonal_patterns=None, percent=10):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'validation']
        type_map = {'train': 0, 'validation': 1, 'test': 2}
        self.set_type = type_map[flag]

        # self.features = features
        # self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # if self.features == 'M' or self.features == 'MS':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        # elif self.features == 'S':
        #    df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            df_stamp['minute'] = df_stamp['date'].dt.minute

            # Additional processing for minute
            df_stamp['minute'] = df_stamp['minute'] // 15

            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            data_stamp = None

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class DatasetCustom(Dataset):
    """
    A dataset class for handling custom datasets for time series forecasting and imputation tasks.

    This dataset supports loading and preprocessing custom datasets for training, testing, and validation purposes. It provides
    functionality to scale the data, encode time features, and specify the percentage of data to use.

    Parameters:
        root_path (str): Root directory where the dataset is stored.
        flag (str): Specifies the dataset split to load. Valid options are 'train', 'test', or 'validation'.
        size (List[int], optional): Specifies the sequence length, label length, and prediction length. Defaults to [24*4*4, 24*4, 24*4].
        data_path (str, optional): Path to the specific dataset file. Defaults to 'ETTh1.csv'.
        scale (bool, optional): If True, scales the data using StandardScaler. Defaults to True.
        timeenc (int, optional): Encoding for time features (0 for no encoding, 1 for encoding). Defaults to 0.
        freq (str, optional): Frequency of the time series data (e.g., 'h' for hourly). Defaults to 'h'.
        percent (int, optional): Percentage of the dataset to use, mainly for testing purposes. Defaults to 10.

    The dataset is expected to be in CSV format with a column named 'date' for timestamps and additional columns for features.
    """
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, timeenc=0, freq='h', percent=10):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'validation']
        type_map = {'train': 0, 'validation': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = 'OT'
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            data_stamp = None

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
