import tensorflow as tf
import pandas as pd
import numpy as np


def example_generator(mode, max_seq_len, dataset_path, tokenizer, input_categories, output_categories):
    if mode == "train":
        # 训练模式
        df_train = pd.read_csv(dataset_path, engine='python', encoding="utf-8")
        df_train_context = df_train[input_categories]   # 训练样本所在列
        df_train_label = df_train[output_categories]    # 标签样本所在列
        print("train shape is ", df_train.shape[0])
        index_list = np.arange(df_train.shape[0])
        np.random.shuffle(index_list)
        for i in index_list:
            inputs_context = str(df_train_context.iloc[i])
            outputs_label = df_train_label.iloc[i]
            inputs = tokenizer.encode_plus(inputs_context,
                                           add_special_tokens=True,
                                           max_length=max_seq_len,
                                           truncation_strategy='longest_first')  # 进行编码

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]

            # label = np.asarray(outputs_label.astype(int) + 1)
            label = outputs_label.astype(int) + 1

            output = {
                "input_ids": input_ids,  # 输入文本串编码ids
                "attention_mask": attention_mask,  # attention mask输入
                "token_type_ids": token_type_ids,  # token type输入
                "label": label
            }
            yield output
    elif mode == "test":
        # 测试阶段
        df_test = pd.read_csv(dataset_path, engine='python', encoding="utf-8")
        df_test_context = df_test[input_categories]  # 训练样本所在列
        print("test shape is ", df_test.shape[0])
        index_list = np.arange(df_test.shape[0])

        for i in index_list:
            inputs_context = str(df_test_context.iloc[i])
            inputs = tokenizer.encode_plus(inputs_context,
                                           add_special_tokens=True,
                                           max_length=max_seq_len,
                                           truncation_strategy='longest_first')  # 进行编码

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]

            # label = np.asarray(outputs_label.astype(int) + 1)
            label = 0

            output = {
                "input_ids": input_ids,  # 输入文本串编码ids
                "attention_mask": attention_mask,  # attention mask输入
                "token_type_ids": token_type_ids,  # token type输入
                "label": label
            }
            yield output


def batch_generator(generator, mode, max_seq_len, batch_size, dataset_path, tokenizer, input_categories,
                    output_categories):
    dataset = tf.data.Dataset.from_generator(lambda: generator(mode, max_seq_len, dataset_path, tokenizer,
                                                               input_categories, output_categories),
                                             output_types={
                                                 "input_ids": tf.int32,
                                                 "attention_mask": tf.int32,
                                                 "token_type_ids": tf.int32,
                                                 "label": tf.int32,
                                             },
                                             output_shapes={
                                                 "input_ids": [None],
                                                 "attention_mask": [None],
                                                 "token_type_ids": [None],
                                                 "label": [],
                                             })

    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=({"input_ids": [None],
                                                   "attention_mask": [None],
                                                   "token_type_ids": [None],
                                                   "label": []}),
                                   padding_values={"input_ids": tokenizer.pad_token_id,
                                                   "attention_mask": 0,
                                                   "token_type_ids": 0,
                                                   "label": -1},
                                   drop_remainder=True)

    # padded_batch参考连接https://blog.csdn.net/cqupt0901/article/details/108030260
    def update(entry):
        return ({"input_ids": entry["input_ids"],
                 "attention_mask": entry["attention_mask"],
                 "token_type_ids": entry["token_type_ids"]},

                {"label": entry["label"]})

    dataset = dataset.map(update)
    return dataset


def batcher(tokenizer, params, dataset_path):
    dataset = batch_generator(example_generator, params["mode"], params["max_sequence_length"],
                              params["batch_size"], dataset_path, tokenizer, params["input_categories"],
                              params["output_categories"])
    return dataset
