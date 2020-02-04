


# In[1]:


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


# In[29]:


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# In[3]:


ds_train = tfds.load(name="cityscapes", split='train')
ds_train = ds_train.shuffle(100).batch(4)
ds_val = tfds.load(name="cityscapes", split='validation')
ds_val = ds_val.batch(4)


# In[4]:


def myfunc(features):
    image = tf.image.resize(features["image_left"], (256, 512))
    label = tf.image.resize(features["segmentation_label"], (256, 512))
    return image, label
ds_train_new = ds_train.map(myfunc)
ds_val_new = ds_val.map(myfunc)


# In[31]:


for features in ds_train_new.take(1):
    image, segmentation = features
print(image.shape, segmentation.shape)


# In[39]:


display([image[0], segmentation[0]])


# In[8]:


inputs = tf.keras.layers.Input(shape=[None, None, 3])
hidden = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
hidden = tf.nn.relu(hidden)
hidden = tf.keras.layers.Conv2D(64, 3, padding='same')(hidden)
hidden = tf.nn.relu(hidden)
# hidden = tf.keras.layers.Conv2D(128, 3, padding='same')(hidden)
# hidden = tf.nn.relu(hidden)
outputs = tf.keras.layers.Conv2D(34, 3, padding='same')(hidden)
Model = tf.keras.Model(inputs=inputs, outputs=outputs)
Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:


tf.keras.utils.plot_model(Model, show_shapes=True)


# In[20]:


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


# In[21]:


def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = Model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


# In[22]:


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions(ds_train_new)
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


# In[23]:


model_history = Model.fit(ds_val_new, epochs=10,
                          steps_per_epoch=500//16,
                          validation_steps=500//16,
                          validation_data=ds_val_new,
                          callbacks=[DisplayCallback()])


# In[41]:


pred = Model.predict(image[1:2])
# pred = np.argmax(pred, axis=-1)
# pred
# display([sample_image, sample_mask,
#              create_mask(model.predict(sample_image[tf.newaxis, ...]))])


# In[42]:


display([image[1:2], segmentation[1:2], pred[..., tf.newaxis][0]])


# In[ ]:




