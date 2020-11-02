import tensorflow as tf
from transformers import BertConfig, TFBertModel


def load_bert(params):
    # 加载Bert模型
    config = BertConfig.from_pretrained(params["bert_pretrain_path"] + 'bert-base-chinese-config.json',
                                        output_hidden_states=True)
    bert_model = TFBertModel.from_pretrained(params["bert_pretrain_path"] + 'bert-base-chinese-tf_model.h5',
                                             config=config)
    return bert_model


class bert_classify(tf.keras.Model):

    def __init__(self, params):
        super(bert_classify, self).__init__()

        self.bert = load_bert(params)
        self.concat = tf.keras.layers.Concatenate(axis=2)
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.linear = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(3, activation="softmax")])

    def call(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        sequence_output, pooler_output, hidden_states = self.bert(input_ids, attention_mask=attention_mask,
                                                                  token_type_ids=token_type_ids)
        h12 = tf.reshape(hidden_states[-1][:, 0], (-1, 1, 768))
        h11 = tf.reshape(hidden_states[-2][:, 0], (-1, 1, 768))
        h10 = tf.reshape(hidden_states[-3][:, 0], (-1, 1, 768))
        h09 = tf.reshape(hidden_states[-4][:, 0], (-1, 1, 768))
        concat_hidden = self.concat(([h12, h11, h10, h09]))
        x = self.global_average_pooling(concat_hidden)
        # x = x + pooler_output
        output = self.linear(x)
        return output


# BERT模型
def create_model(params):
    input_id = tf.keras.layers.Input((params["max_sequence_length"],), dtype=tf.int32)
    input_mask = tf.keras.layers.Input((params["max_sequence_length"],), dtype=tf.int32)
    input_atn = tf.keras.layers.Input((params["max_sequence_length"],), dtype=tf.int32)

    config = BertConfig.from_pretrained(params["bert_pretrain_path"] + 'bert-base-chinese-config.json',
                                        output_hidden_states=False)
    bert_model = TFBertModel.from_pretrained(params["bert_pretrain_path"] + 'bert-base-chinese-tf_model.h5',
                                             config=config)

    sequence_output, pooler_output, hidden_states = bert_model(input_id, attention_mask=input_mask,
                                                               token_type_ids=input_atn)
    # (bs,140,768)(bs,768)
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    # x = x + pooler_output
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn], outputs=x)
    return model
