
# coding: utf-8

# In[1]:


import os
import re
from PIL import Image
import numpy as np
import keras
import pandas as pd
import keras.optimizers as op
initializations = keras.initializers


# In[25]:


#数据采集处理
data_x = []
data_y = []
for root, dirs, files in os.walk('input_data', topdown=False):
    for name in files:
        filepath = os.path.join(root, name)
        image = Image.open(filepath)
        image = image.convert('L')
        image_arr = np.array(image)
        label = re.findall(r'/(.+?)/',filepath)[0]
        data_x.append(image_arr)
        data_y.append(label)
np.savez('data.npz',data_x,data_y)


# In[26]:


#获取数据把数据分为训练集，测试集
data = np.load('data.npz')
X,y = data['arr_0'],data['arr_1']
y = pd.get_dummies(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X/255,y,test_size=0.2,random_state=1)


# In[6]:


#attention网络架构
from keras.layers import Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


# In[78]:


from keras.layers import *
from keras.models import *
MAX_SENT_LENGTH = 64*64
sentence_input = Input(shape=(64,64), dtype='float32')
l_lstm = Bidirectional(GRU(100, return_sequences=True))(sentence_input)
l_dense = TimeDistributed(Dense(200))(l_lstm)  # 对句子中的每个词
l_att = AttentionLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(64,64), dtype='float32')
x = Conv1D(160,2,activation='relu')(review_input)
x = BatchNormalization()(x)
x = MaxPool1D(2)(x)
x = Conv1D(64,2,activation='relu',padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool1D(2)(x)
conv = x.get_shape
x = Reshape((15,-1))(x)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(x)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
#添加自己写的attention层
l_att_sent = AttentionLayer()(l_dense_sent)
l_att_sent = BatchNormalization()(l_att_sent)
preds = Dense(4, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',optimizer=op.Adam(lr=1e-4),
              metrics=['accuracy'])


# In[95]:


model.fit(x_train,y_train,epochs=10)


# In[96]:


model.evaluate(x_test,y_test)

