# coding: UTF-8
import numpy as np
import tensorflow as tf
import logging
import time
import warnings
warnings.filterwarnings('ignore')


# 调用GPU加速
gpus = tf.config.experimental.list_physical_devices(device_type='gpus')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
# 得到文本对应的id
def data_generator(f_path, params):
    with open(f_path, encoding='utf-8') as f:
        print('Reading:', f_path)
        for line in f:
            line = line.rstrip()
            label, text = line.split('\t')
            text = text.split(' ')
            x = [params['word2id'].get(word, len(word2id)) for word in text]
            if len(x) >= params['max_len']:
                x = x[:params['max_len']]
            else:
                x += [0] * (params['max_len'] - len(x))
            y = int(label)
            yield x, y
            
              
# 得到数据集
def dataset(is_training, params):
    _shapes = ([params['max_len']], ())
    _type = (tf.int32, tf.int32)
    
    if is_training:
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(params['train_path'], params),
            output_shapes = _shapes,
            output_types = _type
        )
        ds = ds.shuffle(params['num_samples'])
        ds = ds.batch(params['batch_size'])
    else:
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(params['test_path'], params),
            output_shapes = _shapes,
            output_types = _type
        )
        ds = ds.batch(params['batch_size'])
        
    return ds
        

# 设置参数
params = {
    'vocab_path': './data/vocab/word.txt',
    'train_path': './data/train.txt',
    'test_path': './data/test.txt',
    'num_samples': 25000,
    'num_labels': 2,
    'batch_size': 32,
    'max_len': 1000,
    'rnn_units': 200,
    'dropout_rate': 0.5,
    'clip_norm': 10.,  # 防止梯度过大
    'num_patience': 3,  # 没有下降的最大次数
    'lr': 3e-4  # 学习率
}


class Model(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
        
        self.embedding = tf.Variable(np.load('./data/vocab/word.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False
                                     )
        self.drop1 = tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop2 = tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop3 = tf.keras.layers.Dropout(params['dropout_rate'])
        
        self.rnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        self.rnn3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        
        self.drop_fc = tf.keras.layers.Dropout(params['dropout_rate'])
        self.fc = tf.keras.layers.Dense(2 * params['rnn_units'], tf.nn.elu)
        
        self.out_linear = tf.keras.layers.Dense(2)
        
    def call(self, inputs, training=False):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)
        batch_sz = tf.shape(inputs)[0]
        run_units = 2 * params['rnn_units']
        # inputs形状 是 batch * maxlen，即多少个样本为一组，每个样本的长度固定为最大长度
        x = tf.nn.embedding_lookup(self.embedding, inputs)
        # embedding 之后，多了50维词向量，即x为 batch * maxlen * 50
        x = tf.reshape(x, (batch_sz*10*10, 10, 50))
        x = self.drop1(x, training=training)
        x = self.rnn1(x)
        x = tf.reduce_max(x, 1)
        x = tf.reshape(x, (batch_sz*10, 10, run_units))
        x = self.drop2(x, training=training)
        x = self.rnn2(x)
        x = tf.reduce_max(x, 1)
        x = tf.reshape(x, (batch_sz, 10, run_units))
        x = self.drop3(x, training=training)
        x = self.rnn3(x)
        x = tf.reduce_max(x, 1)
        x = self.drop_fc(x, training=training)
        x = self.fc(x)
        x = self.out_linear(x)
        
        return x
    
    
def is_descending(history: list):
    # 若经过 num_patience 次梯度下降，损失值或者准确率没有优化，反而下降
    # 则返回 true ，停止迭代
    history = history[-(params['num_patience'] + 1):]
    for i in range(1, len(history)):
        if history[i - 1] <= history[i]:
            return False
    return True
    

# 得到新的word2id映射表
word2id = {}
with open(params['vocab_path'], encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.rstrip()
        word2id[line] = i

params['word2id'] = word2id
params['vocab_size'] = len(word2id) + 1


model = Model(params)
model.build(input_shape=(None, None))

decay_lr = tf.optimizers.schedules.ExponentialDecay(params['lr'], 1000, 0.95)
optim = tf.optimizers.Adam(params['lr'])
global_step = 0
history_acc = []
best_acc = 0
t0 = time.time()
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

while True:
    # 训练模型
    for texts, labels in dataset(is_training=True, params=params):
        with tf.GradientTape() as  tape: # 梯度带，记录所有在上文中的操作，并且通过调用 .gradient() 获得任何上下文中计算得出的张量的梯度
            logits = model(texts, training=True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(loss)
            
        optim.lr.assign(decay_lr(global_step))
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, params['clip_norm']) # 将梯度限制一下，有的时候会更新太猛，防止过拟合
        optim.apply_gradients(zip(grads, model.trainable_variables))  # 更新梯度

        if global_step % 50 == 0:
            logger.info(f"Step {global_step} | Loss: {loss.numpy().item():.4f} | \
                        Spent: {time.time() - t0:.1f} secs | \
                        LR: {optim.lr.numpy().item():.6f} ")
            t0 = time.time()
        global_step += 1
        
    # 验证集效果
    m = tf.keras.metrics.Accuracy()
    
    for texts, labels in dataset(is_training=False, params=params):
        logits = model(texts, training=False)
        y_pred = tf.argmax(logits, axis=-1)
        m.update_state(y_true=labels, y_pred=y_pred)
    
    acc = m.result().numpy()
    logger.info(f"Evaluation: Testing Accuracy: {acc:.3f}")
    history_acc.append(acc)
    
    if acc > best_acc:
        best_acc = acc
    logger.info(f"Best Accuracy: {best_acc:.3f}")
    
    if len(history_acc) > params['num_patience'] and is_descending(history_acc):
        logger.info(f"Test Accuracy not improved over {params['num_patience']} epochs , Early stop")
        break
