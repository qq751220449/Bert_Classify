import os
import tensorflow as tf
from transformers import BertTokenizer
from data_utils import batcher
from models import bert_classify
import tensorflow.keras.backend as K
from tqdm import tqdm
import numpy as np
import pandas as pd


def test(params):
    # 加载预训练模型的tokenizer
    tokenizer = BertTokenizer.from_pretrained(params["bert_pretrain_path"]+'bert-base-chinese-vocab.txt')

    test_batch = batcher(tokenizer, params, params["test_dataset_path"])
    df_test = pd.read_csv(params["test_dataset_path"], engine='python', encoding="utf-8")
    df_sub = df_test[['微博id']]

    if params["use_cross_valid"]:
        test_preds = []
        for i in range(params["n_splits"]):
            cross_pred = []
            K.clear_session()
            print("Building the model ...")
            model = bert_classify(params)

            print("Creating the checkpoint manager")
            checkpoint_dir = "{}/cross_valid_checkpoint/ckpt_{}/checkpoint".format(params["model_checkpoint_dir"], i)
            ckpt = tf.train.Checkpoint(BertModel=model)
            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

            ckpt.restore(ckpt_manager.latest_checkpoint)
            if ckpt_manager.latest_checkpoint:
                print("Restored from {}".format(ckpt_manager.latest_checkpoint))
            else:
                print(checkpoint_dir)
                raise Exception("No checkpoint files.")

            for test_batcher in tqdm(test_batch):
                pred = model(test_batcher[0])
                cross_pred.append(pred)
                # print(pred.shape)

            cross_pred = tf.concat(cross_pred, axis=0)
            test_preds.append(cross_pred)
        test_preds = tf.stack(test_preds, axis=0)
        sub = np.average(test_preds, axis=0)
        sub = np.argmax(sub, axis=1)
        df_sub['y'] = sub - 1
        df_sub.columns = ['id', 'y']
        df_sub.to_csv('test_sub.csv', index=False, encoding='utf-8')
    else:
        cross_pred = []
        K.clear_session()
        print("Building the model ...")
        model = bert_classify(params)

        print("Creating the checkpoint manager")
        checkpoint_dir = "{}/split_valid_checkpoint/checkpoint".format(params["model_checkpoint_dir"])
        ckpt = tf.train.Checkpoint(BertModel=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

        ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print(checkpoint_dir)
            raise Exception("No checkpoint files.")

        for test_batcher in tqdm(test_batch):
            pred = model(test_batcher[0])
            cross_pred.append(pred)

        cross_pred = tf.concat(cross_pred, axis=0)
        sub = np.argmax(cross_pred, axis=1)
        df_sub['y'] = sub - 1
        df_sub.columns = ['id', 'y']
        df_sub.to_csv('test_sub.csv', index=False, encoding='utf-8')

