# coding=utf-8
import sys
import os
import tensorflow as tf
from train_helper import train
from test_helper import test
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

sys.path.append(BASE_DIR)

import argparse


def main():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument("--max_sequence_length", default=140, help="Bert input max sequence length", type=int)

    # 路径参数设置
    parser.add_argument("--train_dataset_path", default='{}/dataset/src_data/train_dataset/nCoV_100k_train.labled.csv'.format(BASE_DIR), help="Train folder")
    parser.add_argument("--test_dataset_path", default='{}/dataset/src_data/test_dataset/nCov_10k_test.csv'.format(BASE_DIR), help="Test folder")
    parser.add_argument("--test_submit_example_path", default='{}/data/test_dataset/submit_example.csv'.format(BASE_DIR), help="submit_example folder")
    parser.add_argument("--bert_pretrain_path", default='{}/dataset/bert_base_chinese/'.format(BASE_DIR), help="Bert Pretrain folder")

    # others
    parser.add_argument("--input_categories", default="微博中文内容", help="输入文本的文本内容列")
    parser.add_argument("--output_categories", default="情感倾向", help="标签列")
    parser.add_argument("--epochs", default=2, help="train epochs", type=int)
    parser.add_argument("--batch_size", default=8, help="train batch_size", type=int)

    # 交叉验证参数
    parser.add_argument("--n_splits", default=5, help="train n_splits", type=int)
    parser.add_argument("--use_cross_valid", default=True, help="是否使用交叉验证")
    parser.add_argument("--cross_dataset_path", default='{}/dataset/cross_data/'.format(BASE_DIR),
                        help="Cross valid folder")

    # 数据集分割路径参数
    parser.add_argument("--split_dataset_path", default='{}/dataset/split_data/'.format(BASE_DIR), help="Split dataset folder")

    # mode
    parser.add_argument("--mode", default='test', help="training or test options")
    parser.add_argument("--loss_type", default="focal_loss", help="loss type is focal_loss or cross_entropy")
    parser.add_argument("--learning_rate_1", default=1e-5, help="learning_rate_1")
    parser.add_argument("--learning_rate_2", default=1e-4, help="learning_rate_2 is None or 1e-4...")
    parser.add_argument("--use_different_learning_rate", default=True, help="是否使用不同的学习率")


    # checkpoint
    parser.add_argument("--model_checkpoint_dir", default='{}/ckpt'.format(BASE_DIR), help="Model folder")

    args = parser.parse_args()
    params = vars(args)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

    if gpus:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')

    if params["mode"] == "train":
        train(params)
    elif params["mode"] == "test":
        test(params)


if __name__ == "__main__":
    main()
