import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

input_categories = '微博中文内容'  # 中文文本内容
output_categories = '情感倾向'  # 标签列

# 训练数据集所在目录,使用交叉验证进行训练
src_data = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                        "./dataset/src_data/train_dataset/nCoV_100k_train.labled.csv"))
print("Source training dataset path is: ", src_data)

cross_data = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "./dataset/cross_data/"))
if not os.path.exists(cross_data):
    os.mkdir(cross_data)

df_train = pd.read_csv(src_data, engine='python', encoding="utf-8")
df_train = df_train[df_train[output_categories].isin(['-1', '0', '1'])]
print('train shape =', df_train.shape)  # 存在标签的合适的数据

K_Fold = StratifiedKFold(n_splits=5).split(X=df_train[input_categories].fillna('-1'),
                                           y=df_train[output_categories].fillna('-1'))

# print(K_Fold)
for index, (train_idx, valid_idx) in enumerate(K_Fold):
    cross_train_file_name = os.path.abspath(os.path.join(cross_data, "cross_train_dataset_{}.csv".format(index)))
    cross_valid_file_name = os.path.abspath(os.path.join(cross_data, "cross_valid_dataset_{}.csv".format(index)))
    train_seg = df_train.iloc[train_idx]
    valid_seg = df_train.iloc[valid_idx]
    train_seg.to_csv(cross_train_file_name, index=None, header=True)
    valid_seg.to_csv(cross_valid_file_name, index=None, header=True)

# 直接进行数据集的随机切分,不适用交叉验证
split_data = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "./dataset/split_data/"))
if not os.path.exists(split_data):
    os.mkdir(split_data)

train_data, valid_data = train_test_split(df_train, test_size=0.1, random_state=42)
train_split_dataset_file_name = os.path.abspath(os.path.join(split_data, "train_split_dataset.csv"))
valid_split_dataset_file_name = os.path.abspath(os.path.join(split_data, "valid_split_dataset.csv"))
train_data.to_csv(train_split_dataset_file_name, index=None, header=True)
valid_data.to_csv(valid_split_dataset_file_name, index=None, header=True)


