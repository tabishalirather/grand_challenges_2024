#!/usr/bin/env python
# coding: utf-8

# In[89]:


from threading import activeCount

import numpy as np
# from keras.src.layers import RepeatVector, TimeDistributed44
# noinspection PyUnresolvedReferences

from tensorflow.keras.layers import RepeatVector, TimeDistributed

from torchgen.executorch.api.et_cpp import return_type

from get_data import get_data
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Input, Dense, LeakyReLU, LSTM
# noinspection PyUnresolvedReferences

from tensorflow.keras.models import Model


# In[90]:


ticker = "AMZN"
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
start_date = '2017-01-01'
end_date = '2023-01-01'
seq_train_length = 10
steps_to_predict = 2
scale = True
test_size = 0.2
save_data = False
split_by_date = False

d_r = get_data(
	ticker,
	feature_columns,
	start_date,
	end_date,
	seq_train_length,
	steps_to_predict,
	scale,
	test_size,
	save_data,
	split_by_date
)

print(d_r[0].head(3))
# print(d_r[1].head(3))


#

# In[91]:


result_df = d_r[1]
# print(result_df)

x_train = result_df['X_train']
y_train = result_df['y_train']
x_test = result_df['X_test']
y_test = result_df['y_test']

# print("X_train shape:", result_df['X_train'].shape)
# print("Y_train shape:", result_df['y_train'].shape)
# print("X_test shape:", result_df['X_test'].shape)
# print("Y_test shape:", result_df['y_test'].shape)
#
# print("First 3 elements of X_train:", result_df['X_train'][:3])
# print("First 3 elements of Y_train:", result_df['y_train'][:3])
# print("First 3 elements of X_test:", result_df['X_test'][:3])
# print("First 3 elements of Y_test:", result_df['y_test'][:3])




# In[92]:


num_time_steps = x_train.shape[1]
num_features = x_train.shape[2]
print(num_time_steps, num_features)
auto_encode_input = Input(shape=(num_time_steps, num_features))


# In[93]:


encoder = LSTM(64, return_sequences = True)(auto_encode_input)
encoder = LSTM(32, return_sequences = True)(encoder)
encoder = LSTM(16, return_sequences = False)(encoder)
# encoder = LSTM()(encoder)


# In[94]:


latent_space = Dense(16, activation = "relu")(encoder)



# In[95]:


decoder = RepeatVector(num_time_steps)(latent_space)
decoder = LSTM(32, return_sequences = True)(decoder)
decoder = LSTM(64, return_sequences = True)(decoder)
decoder = TimeDistributed(Dense(num_features, activation = "linear"))(decoder)


# In[96]:


# decoder = Dense(num_features, activation = "sigmoid")(decoder)


# In[97]:


autoecoder = Model(inputs = auto_encode_input, outputs = decoder)
autoecoder.compile(optimizer = "adam", loss = "mse")


# In[98]:


autoecoder.fit(x_train, x_train, epochs = 10, batch_size = 32, validation_data = (x_test, x_test))


# In[99]:


encoder_model = Model(inputs = auto_encode_input, outputs = latent_space)
x_test_pred = autoecoder.predict(x_test)


# In[100]:


encoded_data = encoder_model.predict(x_test)
print(encoded_data.shape)


# In[101]:


print(encoded_data)


# In[102]:


mse = np.mean(np.power(x_test - x_test_pred, 2), axis=(1, 2))
mae = np.mean(np.abs(x_test - x_test_pred), axis=(1, 2))

# Print average reconstruction error
print("Average MSE reconstruction error:", np.mean(mse))
print("Average MAE reconstruction error:", np.mean(mae))


# In[ ]:





# In[105]:


# compressed_x_train = encoder_model.predict(x_train)
# compressed_x_test = encoder_model.predict(x_test)


# In[106]:


import pickle

compressed_x_train = encoder_model.predict(x_train).reshape(x_train.shape[0], -1)
compressed_x_test = encoder_model.predict(x_test).reshape(x_test.shape[0], -1)

# with open("compressed_x_train.pkl", "wb") as f:
#     pickle.dump(compressed_x_train, f)
# with open("compressed_x_test.pkl", "wb") as f:
#     pickle.dump(compressed_x_test, f)

