import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist  # hand written of digit 0 to digit 9 images in 28x28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("size of the x_train is: ", len(x_train))
print(x_train[0])

plt.imshow(x_train[0])
plt.show()

plt.imshow(x_train[0], cmap="binary")
plt.show()

# We want to normalize the data, 0-255 can be normalized to between 0 and 1
# We don't have to, but it affects significantly

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

print("size of the x_train after normalizing is: ", len(x_train))

plt.imshow(x_train[0])
plt.show()

# There is two types of models, sequential is the most common one. Feed-forward style.

model = tf.keras.models.Sequential()

# And the model add syntax. The first layer will be input layer. Our images are 28x28, multidimensional.
# We can use numpy.reshape or we can use Keras flatten function()
# Flatten is more than just input layer. We can use it at densely connected layers too.
model.add(tf.keras.layers.Flatten())

# Now, we added our input layer. We have to add more layer.
# We can use just 2 hidden layer. This is not too complex to solve.
# The keyword for hidden layer is "Dense". It can take some parameters
# One of the parameters is how many units in the layer, in other words "neurons".
# And then we have to pass the activation function. "Stepper", "Binary", "Sigmoid, "ReLU"
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# We can just simple copy and paste the same line for the second hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Now we have to add the final layer which is output layer, which is a still dense layer but not 128
# The output layer will always be the size of the classification
# In our case it is 10, and we don't want the activation function to be RELU.
# Because it's a probability distribution. So we want to use "softmax"

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# That's our entire model, we're done. We have defined the architecture of our model
# New we need to define is some parameters for the training of the model.
# We got a keyword named "compile" which configures the model for training.
# We're gonna pass some parameters. First one is "optimizer" the want to use.
# We're gonna pass loss metric. Loss is a degree of error. It's basically what you got wrong.
# So normal neural network doesn't actually attempt to optimize for accuracy.
# It doesn't try to maximize the accuracy. It always try to minimize loss.
# So the way you calculate the loss can make a huge impact
# Because it is what is losses relationship to your accuracy
# The optimizer we're going to use is going to be "Adam" optimizer.
# Why did we use "Adam". Could we use other optimizers? What are they?
# This is like the most complex part of the entire neural network.
# So if you are familiar with the "gradiant descent", you could pass something like "stochastic gradient descent"
# But "Adam" optimizer is kind of like rectified linear, kind of the default go-to optimizer
# We could use others, there are not lots of them, ten or so in Keras.

# To start with the loss again, there are many ways to calculate loss
# Probably the most popular one is "categorical_crossentropy" or some version of that
# In this case, we're gonna use "sparse_categorical_crossentropy".
# You can also use binary like in the cases of cats versus dogs.
# Finally what are the metrics we want to track like as we go and we are going to just do accuracy

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# So we have all this and we're actually ready to train the model
# So to train the keyword is "fit" and we need to pass what do we want to fit
# In this case our model is "x_train" and "y_train" and epochs 3. An epoch is a full iteration over samples
# In this case epoch=3 means the model will be trained 3 times in row

model.fit(x_train, y_train, epochs=3)

# This was in sample. So this might this is always gonna really excite you.
# What what's really important to remember is neural networks are great at fitting
# The question is did they over fit? So the idea or hope is that your model actually generalized
# It learnt patterns and in actual attributes to the what makes an eight than what makes an four
# Rather than memorizing every single sample you passed and you'd be surprised how easily a model
# can just memorize all the samples that you passed to do very well.
# So next thing we always do is calculate the validation loss and validation accuracy.
# To do this keyword is evaluate.

val_loss, val_accuracy = model.evaluate(x_test, y_test)
print(val_loss, val_accuracy)

# You should expect your out-of-sample accuracy to be slightly lower and your loss to be slightly higher
# What you definitely don't want to see is either too close or too much of delta
# If there is a huge Delta, chances are you probably
# already have over fit and you'd want to like kind of dial it back a little bit
# That's basically everything.
# If you want to save a model load a module
# To save it, model.save('epic_num_reader.model')
# To load it, new_model = tf.keras.models.load_model('epic_num_reader_model')

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')

# and finally if we want to make predictions the keyword we need to use is "predict"

predictions = new_model.predict([x_test])

# If we just print predictions, it's probably not gonna look too friendly
print(predictions)

# What's going on here? These are all one hot arrays. These are our probability distributions
# So what do we do with these, we're gonna use numpy, you can use tf.arcmax but it's abstract
# It's a tensor and we have to pull it down, we need a session, it is just easier to import numpy

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()
