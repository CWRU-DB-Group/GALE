from layers import *
from metrics import *



flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.precision =0.0
        self.recall = 0.0
        self.f1_score = 0.0


    def _build(self):
        raise NotImplementedError

    def build(self, reuse=False):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name, reuse=reuse):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        ##output the intermediate features generate by the first gcn layer
        self.features = self.activations[1]
        self.outputs = self.activations[-1]

        # Store model variables for easy access

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}




    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def _precision(self):
        raise NotImplementedError

    def _f1_score(self):
        raise NotImplementedError

    def _recall(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)



class GAN(Model):
    """
    A GAN model.
    :param real_size: The shape of the real data.
    :param z_size: The number of entries in the z code vector.
    :param learnin_rate: The learning rate to use for Adam.
    :param num_classes: The number of classes to recognize.
    :param alpha: The slope of the left half of the leaky ReLU activation
    :param beta1: The beta1 parameter for Adam.
    """



    def __init__(self, placeholders,real_input_dim, z_input_dim, alpha=0.2, beta1= 0.5, **kwargs):
        super(GAN, self).__init__(**kwargs)

        self.learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
        self.input_real = placeholders['features']
        self.input_real_dim = real_input_dim
        self.input_z = placeholders['inputs_z']
        self.input_z_dim = z_input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.y = placeholders['labels']
        self.label_mask = placeholders['labels_mask']
        self.dropout = placeholders['dropout']
        self.placeholders = placeholders


        loss_results = model_loss(self.input_real, self.input_z,
                                  self.input_real_dim, self.y, self.output_dim,
                                  self.placeholders,
                                  label_mask=self.label_mask,
                                  alpha=0.2,
                                  drop_rate=self.dropout
                                  )
        self.real_data_loss, self.fake_data_loss,self.unsupervised_loss, self.d_model_labeled_data, self.d_loss, self.g_loss, self.correct, self.masked_correct, self.samples, \
            self.masked_pred_class, self.masked_labels, self.data_features= loss_results

        self.d_opt, self.g_opt, self.shrink_lr, self.d_gradients = model_opt(self.d_loss, self.g_loss, self.learning_rate, beta1 =0.5)

def generator(z, output_dim, reuse=False, alpha=0.2, training=True, size_mult=16):
    with tf.variable_scope('generator', reuse=reuse):
        # First fully connected layer
        # modified and achieved good performance
        x1 = tf.layers.dense(z, 4 * 4 * size_mult * 2)
        #x1 = tf.layers.dense(z, 4 * 4 * size_mult * 4)
        # Reshape it to start the convolutional stack
        # modified and achieved good performance
        x1 = tf.reshape(x1, (-1, 8, 8, 1))
        #x1 = tf.reshape(x1, (-1, 4, 4, size_mult * 4))

        # print(x1.shape)
        x1 = tf.layers.batch_normalization(x1, training=training)
        # print(x1.shape)
        x1 = tf.maximum(alpha * x1, x1)
        print(x1.shape)

        # strides can be treated as amplifying numbers, here is 2*times input shape
        # modified and achieved good performance


        x2 = tf.layers.conv2d_transpose(x1, 1, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.maximum(alpha * x2, x2)


        #x2 = tf.layers.conv2d_transpose(x1, size_mult * 2, 5, strides=2, padding='same')
        #x2 = tf.layers.conv2d_transpose(x1, size_mult * 2, 10, strides=2, padding='same')

        #x2 = tf.layers.conv2d_transpose(x1, 1, 3, strides=2, padding='same')
        #x2 = tf.layers.batch_normalization(x2, training=training)
        #x2 = tf.maximum(alpha * x2, x2)

        print(x2.shape)





        #x3 = tf.layers.conv2d_transpose(x2, size_mult, 5, strides=2, padding='same')
        #x3 = tf.layers.conv2d_transpose(x2, size_mult, 10, strides=2, padding='same')
        #x3 = tf.layers.batch_normalization(x3, training=training)
        #x3 = tf.maximum(alpha * x3, x3)




        # Output layer

        x4 = tf.layers.conv2d_transpose(x2, 1, 5, strides=2, padding='same')
        #x4 = tf.layers.conv2d_transpose(x2, 1, 3, strides=2, padding='same')
        #print(x4.shape)
        x4 = tf.reshape(x4, (-1, 32*32))

        logits = tf.layers.dense(x4, output_dim)

        '''
        #x4 = tf.layers.conv2d_transpose(x3, 1, 5, strides=2, padding='same')
        x4 = tf.layers.conv2d_transpose(x3, size_mult, 10, strides=2, padding='same')
        x4 = tf.reshape(x4, (-1, 32 * 32))
        logits = tf.layers.dense(x4, output_dim)
        '''


        out = tf.tanh(logits)
        print(out.shape)
        return out


def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and biases to update. Get them separately for the discriminator and the generator
    discriminator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    print("discriminator_train_vars are: ")
    print("////////////Debug purpose, discriminator variables/////////////")
    for discriminator_train_var in discriminator_train_vars:
        print (discriminator_train_var)
    num_discriminator_train_vars = np.sum([np.prod(v.get_shape().as_list()) for v in discriminator_train_vars])
    print("///////////Debug purpose, discriminator variables//////////////")
    print("total number of trainable discriminator vars {}".format(num_discriminator_train_vars))

    generator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    print("genarator_train_vars are: ")
    print("/////////////Debug purpose, generator variables////////////////")
    for generator_train_var in generator_train_vars:
        print (generator_train_var)
    num_generator_train_vars=np.sum([np.prod(v.get_shape().as_list()) for v in generator_train_vars])
    print("/////////////Debug purpose, generator variables////////////////")
    print("total number of trainable generator vars {}".format(num_generator_train_vars))


    # Minimize both players' costs simultaneously
    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, name='d_optimizer').minimize(d_loss,
                                                                                            var_list=discriminator_train_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, name='g_optimizer').minimize(g_loss,
                                                                                            var_list=generator_train_vars)

    d_train_optimizer = tf.train.AdamOptimizer(learning_rate, beta1, name='d_optimizer')
    d_gradients = d_train_optimizer.compute_gradients(d_loss,
                                                var_list=discriminator_train_vars)
    shrink_lr = tf.assign(learning_rate, learning_rate * 0.9)

    return d_train_opt, g_train_opt, shrink_lr, d_gradients

def model_loss(input_real, input_z, output_dim, y, num_classes, placeholders, label_mask, alpha=0.2, drop_rate=0.,
               smooth=0.1):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input random noise vector
    :param output_dim: The number of channels in the output image
    :param y: Integer class labels
    :param num_classes: The number of classes
    :param alpha: The slope of the left half of leaky ReLU activation
    :param drop_rate: The probability of dropping a hidden unit
    :return: A tuple of (discriminator loss, generator loss)
    """




    # These numbers multiply the size of each layer of the generator and the discriminator,
    # respectively. You can reduce them to run your code faster for debugging purposes.
    g_size_mult = 2
    d_size_mult = 64
    back_label_mask =label_mask

    # Here we instatiate the generator and the discriminator networks
    g_model = generator(input_z, output_dim, alpha=alpha, size_mult=g_size_mult)



    d_on_data = discriminator(input_real, alpha=alpha, drop_rate=drop_rate, size_mult=d_size_mult)

    # d_model_real: probability that the input is real
    # class_logits_on_data: the unnormalized log probability values for the probability of each classe
    # gan_logits_on_data: the probability of whether or not the image is real
    # data_features: features from the last layer of the discriminator to be used in the feature matching loss
    d_model_real, class_logits_on_data, gan_logits_on_data, data_features = d_on_data



    d_on_samples = discriminator(g_model, reuse=True, alpha=alpha, drop_rate=drop_rate, size_mult=d_size_mult)
    d_model_fake, class_logits_on_samples, gan_logits_on_samples, sample_features = d_on_samples


    # Here we compute `d_loss`, the loss for the discriminator.
    # This should combine two different losses:
    # 1. The loss for the GAN problem, where we minimize the cross-entropy for the binary
    #    real-vs-fake classification problem.
    real_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_data,
                                                                            labels=tf.zeros_like(
                                                                                gan_logits_on_data) * (
                                                                                               1 - smooth)))

    fake_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_samples,
                                                                            labels=tf.ones_like(
                                                                                gan_logits_on_samples)* (
                                                                                               1 - smooth)))



    unsupervised_loss = real_data_loss + fake_data_loss
    #######################################




    #  2. The loss for the SVHN digit classification problem, where we minimize the cross-entropy
    #     for the multi-class softmax. For this one we use the labels. Don't forget to ignore
    #     use `label_mask` to ignore the examples that we are pretending are unlabeled for the
    #     semi-supervised learning problem.
    y = tf.squeeze(y)
    #print(y.shape)
    suppervised_loss = tf.nn.softmax_cross_entropy_with_logits(logits=class_logits_on_data,
                                                               labels=y)

    label_mask = tf.squeeze(tf.to_float(label_mask))

    # ignore the labels that we pretend does not exist for the loss
    suppervised_loss = tf.reduce_sum(tf.multiply(suppervised_loss, label_mask))

    # get the mean
    # This part is similar to how gcn generates their suppervised_loss
    suppervised_loss = suppervised_loss / tf.maximum(1.0, tf.reduce_sum(label_mask))
    d_loss = (unsupervised_loss + suppervised_loss)

    # Here we set `g_loss` to the "feature matching" loss invented by Tim Salimans at OpenAI.
    # This loss consists of minimizing the absolute difference between the expected features
    # on the data and the expected features on the generated samples.
    # This loss works better for semi-supervised learning than the tradition GAN losses.

    # Make the Generator output features that are on average similar to the features
    # that are found by applying the real data to the discriminator
    #print("data_feature is")
    #print(data_features)
    data_moments = tf.reduce_mean(data_features, axis=0)
    #print("sample feature is")
    #print(sample_features)
    sample_moments = tf.reduce_mean(sample_features, axis=0)
    g_loss = tf.reduce_mean(tf.abs(data_moments - sample_moments))

    pred_class = tf.cast(tf.argmax(class_logits_on_data, 1), tf.int64)
    eq = tf.equal(tf.argmax(y, 1), pred_class)
    correct = tf.reduce_sum(tf.to_float(eq))
    masked_correct = tf.reduce_sum(label_mask * tf.to_float(eq))


    ####output the precision and recall
    #consider the label_mask first
    pred_class.set_shape([None])
    back_label_mask.set_shape([None])
    masked_pred_class = tf.boolean_mask(pred_class, back_label_mask )
    masked_labels = tf.boolean_mask(y, back_label_mask )

    #output the labeled dataset distribution
    d_model_labeled_data = tf.boolean_mask(d_model_real,back_label_mask )


    return  real_data_loss, fake_data_loss, unsupervised_loss, d_model_labeled_data, d_loss, g_loss, correct, masked_correct, g_model, masked_pred_class, masked_labels, data_features


def discriminator(x, reuse=False, alpha=0.2, drop_rate=0.2, num_classes=2, size_mult=64):
    with tf.variable_scope('discriminator', reuse=reuse):

        print(x.shape)
        x0 = tf.layers.dense(x, 32*32*3)

        x0 = tf.layers.dropout(x0, rate=drop_rate/2.5)
        x0 = tf.reshape(x0, (-1, 32, 32, 3))
        print(x0.shape)

        # Input layer is ?x32x32x3
        x1 = tf.layers.conv2d(x0, size_mult, 3, strides=2, padding='same')
        #x1 = tf.layers.conv2d(x0, size_mult, 5, strides=2, padding='same')


        relu1 = tf.maximum(alpha * x1, x1)
        relu1 = tf.layers.dropout(relu1, rate=drop_rate)  # [?x16x16x?]


        print(relu1.shape)
        x2 = tf.layers.conv2d(relu1, size_mult, 3, strides=2, padding='same')
        #x2 = tf.layers.conv2d(relu1, size_mult, 5, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=True)  # [?x8x8x?]
        relu2 = tf.maximum(alpha * bn2, bn2)

        print(relu2.shape)
        x3 = tf.layers.conv2d(relu2, size_mult, 3, strides=2, padding='same')  # [?x4x4x?]
        #x3 = tf.layers.conv2d(relu2, size_mult, 5, strides=2, padding='same')
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)
        relu3 = tf.layers.dropout(relu3, rate=drop_rate)
        print(relu3.shape)


        x4 = tf.layers.conv2d(relu3, 2 * size_mult, 3, strides=1, padding='same')  # [?x4x4x?]
        bn4 = tf.layers.batch_normalization(x4, training=True)
        relu4 = tf.maximum(alpha * bn4, bn4)
        print(relu4.shape)




        x5 = tf.layers.conv2d(relu4, 2 * size_mult, 3, strides=1, padding='same')  # [?x4x4x?]
        bn5 = tf.layers.batch_normalization(x5, training=True)
        relu5 = tf.maximum(alpha * bn5, bn5)
        print(relu5.shape)



        x6 = tf.layers.conv2d(relu5, 2 * size_mult, 3, strides=2, padding='same')  # [?x2x2x?]
        bn6 = tf.layers.batch_normalization(x6, training=True)
        relu6 = tf.maximum(alpha * bn6, bn6)
        relu6 = tf.layers.dropout(relu6, rate=drop_rate)
        print(relu6.shape)



        x7 = tf.layers.conv2d(relu5, filters=(2 * size_mult), kernel_size=3, strides=1, padding='valid')
        # Don't use bn on this layer, because bn would set the mean of each feature
        # to the bn mu parameter.
        # This layer is used for the feature matching loss, which only works if
        # the means can be different when the discriminator is run on the data than
        # when the discriminator is run on the generator samples.
        relu7 = tf.maximum(alpha * x7, x7)
        print(relu7.shape)

        # Flatten it by global average pooling
        # In global average pooling, for every feature map we take the average over all the spatial
        # domain and return a single value
        # In: [BATCH_SIZE,HEIGHT X WIDTH X CHANNELS] --> [BATCH_SIZE, CHANNELS]
        features = tf.reduce_mean(relu7, axis=[1, 2])
        print("feature shape")
        print(features.shape)

        # Set class_logits to be the inputs to a softmax distribution over the different classes
        class_logits = tf.layers.dense(features, num_classes)
        #This method is a simple method, compared with paper GraphSGAN that generates an additional logits only for fake examples

        # This function is more numerically stable than log(sum(exp(input))).
        # It avoids overflows caused by taking the exp of large inputs and underflows
        # caused by taking the log of small inputs.
        gan_logits = tf.reduce_logsumexp(class_logits, 1)
        print(gan_logits.shape)

        # Get the probability that the input is real rather than fake
        out = tf.nn.softmax(class_logits)  # class probabilities for the 10 real classes plus the fake class

        return out, class_logits, gan_logits, features