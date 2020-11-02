import os
import tensorflow as tf
from transformers import BertTokenizer
from data_utils import batcher
from models import create_model, bert_classify
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import tensorflow.keras.backend as K
from tqdm import tqdm
import time


def train(params):
    # 加载预训练模型的tokenizer
    tokenizer = BertTokenizer.from_pretrained(params["bert_pretrain_path"]+'bert-base-chinese-vocab.txt')

    # 加载训练数据集
    if params["use_cross_valid"]:    # 使用交叉验证技术
        acc_list = []
        for i in range(params["n_splits"]):
            acc_list_split = []
            cross_train_file_name = os.path.abspath(os.path.join(params["cross_dataset_path"], "cross_train_dataset_{}.csv".format(i)))
            cross_valid_file_name = os.path.abspath(os.path.join(params["cross_dataset_path"], "cross_valid_dataset_{}.csv".format(i)))

            train_batch = batcher(tokenizer, params, cross_train_file_name)
            valid_batch = batcher(tokenizer, params, cross_valid_file_name)

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
                print("Initializing from scratch.")

            optimizer1 = tf.keras.optimizers.Adam(learning_rate=params["learning_rate_1"])
            optimizer2 = tf.keras.optimizers.Adam(learning_rate=params["learning_rate_2"])
            loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

            # 定义损失函数
            def loss_function(real, pred):
                real = real["label"]
                real_onehot = tf.one_hot(real, depth=3)
                loss = tf.reduce_sum(loss_object(real_onehot, pred))
                return loss

            def focal_loss(real, pred, alpha=0.5, gamma=2.0):
                # 解决难易样本的问题
                real = real["label"]
                real_onehot = tf.one_hot(real, depth=3)
                pred += tf.keras.backend.epsilon()
                ce = -real_onehot * tf.math.log(pred)
                weight = tf.pow(1 - pred, gamma) * real_onehot
                fl = ce * weight * alpha
                reduce_fl = tf.keras.backend.max(fl, axis=-1)
                return tf.reduce_sum(reduce_fl)

            # @tf.function()
            def train_step(batcher):
                # loss = 0
                with tf.GradientTape(persistent=True) as tape:  # 所有的计算在此处进行
                    pred = model(batcher[0])
                    # loss = loss_function(batcher[1], pred)
                    if params["loss_type"] == "focal_loss":
                        loss = focal_loss(batcher[1], pred)
                    elif params["loss_type"] == "cross_entropy":
                        loss = loss_function(batcher[1], pred)
                bert_variables = model.bert.trainable_variables
                liner_variables = model.concat.trainable_variables + model.global_average_pooling.trainable_variables + model.linear.trainable_variables
                if params["use_different_learning_rate"]:
                    gradients = tape.gradient(loss, bert_variables)
                    optimizer1.apply_gradients(zip(gradients, bert_variables))

                    gradients = tape.gradient(loss, liner_variables)
                    optimizer2.apply_gradients(zip(gradients, liner_variables))
                else:
                    variables = model.trainable_variables
                    # variables = bert_variables + liner_variables
                    gradients = tape.gradient(loss, variables)
                    optimizer1.apply_gradients(zip(gradients, variables))

                return loss

            best_loss = 20
            epochs = params['epochs']
            for epoch in tqdm(range(epochs)):
                t0 = time.time()
                step = 0
                total_loss = 0
                for batch in train_batch:
                    loss = train_step(batch)
                    step += 1
                    total_loss += loss
                    if step % 10 == 0:
                        print('Epoch {} Batch {} Loss {:.6f}'.format(epoch + 1, step, total_loss / step))
                        # print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, loss.numpy()))

                    if step % 2000 == 0:
                        total, total_correct = 0., 0
                        for valid_batcher in valid_batch:
                            pred = model(valid_batcher[0])
                            pred = tf.argmax(pred, axis=1)
                            pred = tf.cast(pred, dtype=tf.int32)
                            correct = tf.equal(pred, valid_batcher[1]["label"])
                            total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                            total += valid_batcher[1]["label"].shape[0]
                        print("step:", step, 'Evaluate Acc:', total_correct / total)
                        acc_list_split.append(("step:" + str(step)))
                        acc_list_split.append(total_correct / total)

                if epoch % 1 == 0:
                    if total_loss / step < best_loss:
                        best_loss = total_loss / step
                        ckpt_save_path = ckpt_manager.save()
                        print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path,
                                                                                          best_loss))
                        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                        print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))

                    total, total_correct = 0., 0
                    for valid_batcher in valid_batch:
                        pred = model(valid_batcher[0])
                        pred = tf.argmax(pred, axis=1)
                        pred = tf.cast(pred, dtype=tf.int32)
                        correct = tf.equal(pred, valid_batcher[1]["label"])
                        total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                        total += valid_batcher[1]["label"].shape[0]
                    print("epoch:", epoch, 'Evaluate Acc:', total_correct / total)
                    acc_list_split.append(("epoch:" + str(epoch)))
                    acc_list_split.append(total_correct / total)
            acc_list.append(acc_list_split)
        print(acc_list)
        acc_txt_path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "./result/"))
        if not os.path.exists(acc_txt_path):
            os.mkdir(acc_txt_path)
        with open(os.path.abspath(os.path.join(acc_txt_path, "test.txt")), "w") as f:
            f.write(str(acc_list))  # 这句话自带文件关闭功能，不需要再写f.close()

    else:
        # 不适用交叉验证技术
        acc_list = []
        split_train_file_name = os.path.abspath(os.path.join(params["split_dataset_path"], "train_split_dataset.csv"))
        split_valid_file_name = os.path.abspath(os.path.join(params["split_dataset_path"], "valid_split_dataset.csv"))
        train_batch = batcher(tokenizer, params, split_train_file_name)
        valid_batch = batcher(tokenizer, params, split_valid_file_name)

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
            print("Initializing from scratch.")

        optimizer1 = tf.keras.optimizers.Adam(learning_rate=params["learning_rate_1"])
        optimizer2 = tf.keras.optimizers.Adam(learning_rate=params["learning_rate_2"])
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        # 定义损失函数
        def loss_function(real, pred):
            real = real["label"]
            real_onehot = tf.one_hot(real, depth=3)
            loss = tf.reduce_sum(loss_object(real_onehot, pred))
            return loss

        def focal_loss(real, pred, alpha=0.5, gamma=2.0):
            # 解决难易样本的问题
            real = real["label"]
            real_onehot = tf.one_hot(real, depth=3)
            pred += tf.keras.backend.epsilon()
            ce = -real_onehot * tf.math.log(pred)
            weight = tf.pow(1 - pred, gamma) * real_onehot
            fl = ce * weight * alpha
            reduce_fl = tf.keras.backend.max(fl, axis=-1)
            return tf.reduce_sum(reduce_fl)

        # @tf.function()
        def train_step(batcher):
            # loss = 0
            with tf.GradientTape(persistent=True) as tape:  # 所有的计算在此处进行
                pred = model(batcher[0])
                # loss = loss_function(batcher[1], pred)
                if params["loss_type"] == "focal_loss":
                    loss = focal_loss(batcher[1], pred)
                elif params["loss_type"] == "cross_entropy":
                    loss = loss_function(batcher[1], pred)
            bert_variables = model.bert.trainable_variables
            liner_variables = model.concat.trainable_variables + model.global_average_pooling.trainable_variables + model.linear.trainable_variables
            print(model.concat.trainable_variables)
            if params["use_different_learning_rate"]:
                gradients = tape.gradient(loss, bert_variables)
                optimizer1.apply_gradients(zip(gradients, bert_variables))

                gradients = tape.gradient(loss, liner_variables)
                optimizer2.apply_gradients(zip(gradients, liner_variables))
            else:
                variables = model.trainable_variables
                variables = bert_variables + liner_variables
                gradients = tape.gradient(loss, variables)
                optimizer1.apply_gradients(zip(gradients, variables))
            return loss

        best_loss = 20
        epochs = params['epochs']
        for epoch in tqdm(range(epochs)):
            t0 = time.time()
            step = 0
            total_loss = 0
            for batch in train_batch:
                loss = train_step(batch)
                step += 1
                total_loss += loss
                if step % 10 == 0:
                    print('Epoch {} Batch {} Loss {:.6f}'.format(epoch + 1, step, total_loss / step))
                    # print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, loss.numpy()))

                if step % 2000 == 0:
                    total, total_correct = 0., 0
                    for valid_batcher in valid_batch:
                        pred = model(valid_batcher[0])
                        pred = tf.argmax(pred, axis=1)
                        pred = tf.cast(pred, dtype=tf.int32)
                        correct = tf.equal(pred, valid_batcher[1]["label"])
                        total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                        total += valid_batcher[1]["label"].shape[0]
                    print("step:", step, 'Evaluate Acc:', total_correct / total)
                    acc_list.append(("step:" + str(step)))
                    acc_list.append(total_correct / total)

            if epoch % 1 == 0:
                if total_loss / step < best_loss:
                    best_loss = total_loss / step
                    ckpt_save_path = ckpt_manager.save()
                    print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path,
                                                                                      best_loss))
                    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                    print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))

                total, total_correct = 0., 0
                for valid_batcher in valid_batch:
                    pred = model(valid_batcher[0])
                    pred = tf.argmax(pred, axis=1)
                    pred = tf.cast(pred, dtype=tf.int32)
                    correct = tf.equal(pred, valid_batcher[1]["label"])
                    total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                    total += valid_batcher[1]["label"].shape[0]
                print("epoch:", epoch, 'Evaluate Acc:', total_correct / total)
                acc_list.append(("epoch:" + str(epoch)))
                acc_list.append(total_correct / total)
        print(acc_list)
        acc_txt_path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "./result/"))
        if not os.path.exists(acc_txt_path):
            os.mkdir(acc_txt_path)
        with open(os.path.abspath(os.path.join(acc_txt_path, "test.txt")), "w") as f:
            f.write(str(acc_list))  # 这句话自带文件关闭功能，不需要再写f.close()

