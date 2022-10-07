from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utils import *
from models import GAN
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from load_fake_data import *
from load_clean_data import *
import timeit
import copy
from sklearn.cluster import KMeans

from datetime import datetime
from shutil import copyfile
import os
from logger import Logger



'''
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join("./results/", run_id)
os.mkdir(run_dir)
copyfile(__file__,os.path.join(run_dir,os.path.basename(__file__)))
'''



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gan', 'Model string.')
flags.DEFINE_float('learning_rate', 5e-4, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1',16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
#Need to be changed by dataset
flags.DEFINE_string('datasetname', 'dm_error', 'Dataset to be used.')
flags.DEFINE_string('datapath',"dataset/dm/","Dataset path")
flags.DEFINE_integer('active_epochs', 17, 'Number of active epochs to continue training discriminator.')
flags.DEFINE_integer('sample_size',70, 'sample size per epoch in active learning')
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_string('sample_method', 'approximation', 'sampling method in active learing part')
flags.DEFINE_integer('query_epoch', 2, 'use to select a query strategy, default set as 5')
flags.DEFINE_integer('query_sample_size', 50, 'use to select a query strategy, default set as 10')
flags.DEFINE_integer('cluster_size',25, 'clustering size')
flags.DEFINE_float("lamda", 0.0005, "lamda in the approximation algo.")
flags.DEFINE_boolean('pca_flag',False,'whether to use PCA as preprocessing')

# Set random seed
seed=FLAGS.seed
np.random.seed(seed)
tf.set_random_seed(seed)

sys.stdout = Logger(FLAGS.datasetname)
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join("./results/", run_id +FLAGS.datasetname)
os.mkdir(run_dir)
print(os.path.basename(__file__))
copyfile(__file__,os.path.join(run_dir,os.path.basename(__file__)))

def calc_scores_error_detection(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    #print("y_true's shape is"+ str(y_true.shape))
    #print("y_pred's shape is"+ str(y_pred.shape))
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    balanced_accuracy_score=metrics.accuracy_score(y_true, y_pred,normalize=True)
    return precision[-1], recall[-1], fscore[-1], support[-1],balanced_accuracy_score


#start marking down the time
start = timeit.default_timer()

learning_rate = FLAGS.learning_rate
#print("learning rate %14.6f"%learning_rate)
for attr,flag_obj in FLAGS.__flags.items():
    print("attr:%s\tvalue:%s" % (attr, flag_obj.value))

with open(FLAGS.datasetname+'gedet++_test.txt', 'a') as f:
    f.write("\n")
    for attr, flag_obj in FLAGS.__flags.items():
        f.write("{0}, {1} \n".format(attr, flag_obj.value))


pca_flag = FLAGS.pca_flag

#load the preprocessed fake data and polluted node ids
feats_fake = load_fake_data(FLAGS.datapath,FLAGS.datasetname,pca_flag )
print(feats_fake.shape)


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_train_active, train_active_mask, ground_truth_train_active, ground_truth_train_active_mask= load_clean_data(FLAGS.datapath,FLAGS.datasetname,pca_flag)

# Some preprocessing
features = preprocess_features(features)
z_dim =features.shape[1]
if FLAGS.model =='gan':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GAN
    z_dim = z_dim
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))


# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32,shape=features.shape),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32) , # helper variable for sparse dropout
    'inputs_z': tf.placeholder(tf.float32, shape=(None, z_dim)) #placehoder for the inputing fake data into discriminator
}


# Create model

model = model_func(placeholders, real_input_dim=features.shape[1], z_input_dim=z_dim, logging=True)


saver = tf.train.Saver(max_to_keep=FLAGS.active_epochs)


checkpoint_path = FLAGS.datasetname+"training_1/"
#checkpoint_path = FLAGS.datasetname + "training_1/"+'90%error_rate/'
checkpoint_dir = os.path.dirname(checkpoint_path)

sample_z = feats_fake
train_d_loss, val_d_loss =  [], []
train_g_loss =[]
f1_score_train=[]

############################################################Real implementation 
# Initialize session
sess = tf.Session()


# Define model evaluation function


def evaluate(features, support, labels, mask, sample_z, placeholders):
    t_test = time.time()
    num_examples = mask.sum()
    num_correct = 0
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    feed_dict_val.update({placeholders['inputs_z']: sample_z})
    val_correct,masked_v_pre, masked_v_label, v_discriminator_loss, d_on_active_data, data_embeds, generator_unsupervised_loss= sess.run([model.masked_correct,model.masked_pred_class, model.masked_labels,model.d_loss,model.d_model_labeled_data, model.data_features, model.unsupervised_loss], feed_dict=feed_dict_val)
    num_correct += val_correct
    val_accuracy = num_correct / float(num_examples)
    y_true = np.argmax(masked_v_label, axis=1)
    y_pred = masked_v_pre
    v_precision, v_recall, v_f1_score, v_support = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)

    return val_accuracy, v_precision[-1], v_recall[-1], v_f1_score[-1], v_discriminator_loss,(time.time() - t_test),d_on_active_data, data_embeds


'''
# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
f1_score_val=[]

best = 0.0




best_epoch = 0
bad_counter = 0

test_f1_max =0.0
t_total = time.time()
# Train model
for epoch in range(FLAGS.epochs):

    print("Epoch",epoch)
    t1e = time.time()
    num_examples = train_mask.sum()
    num_correct = 0
    # Construct feed dictionary for the training part
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['inputs_z']: sample_z})

    _, _, correct, masked_t_pre, masked_t_label,discriminator_loss, generator_loss =sess.run([model.d_opt, model.g_opt, model.masked_correct,
                                                                  model.masked_pred_class, model.masked_labels,model.d_loss, model.g_loss] , feed_dict=feed_dict)
    num_correct += correct
    ##add the training discriminator loss
    train_d_loss.append(discriminator_loss)
    train_g_loss.append(generator_loss)
    sess.run([model.shrink_lr])
    train_accuracy = num_correct / float(num_examples)
    y_true = np.argmax(masked_t_label, axis=1)
    y_pred = masked_t_pre
    # print("y_true's shape is"+ str(y_true.shape))
    # print("y_pred's shape is"+ str(y_pred.shape))
    t_precision, t_recall, t_f1_score, t_support = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    print("\t\tClassifier train accuracy: ", train_accuracy)
    t_accu = metrics.accuracy_score(y_true, y_pred, normalize=True)
    print("\t\tClassifier train accuracy validation:", t_accu)
    print("\t\tClassifier train precision:", t_precision[-1])
    print("\t\tClassifier train recall:", t_recall[-1])
    print("\t\tClassifier train f1_score:", t_f1_score[-1])
    
    # Training step  is finished



    # Validation
    v_acc, v_pre, v_recall, v_f1, v_discriminator_loss, duration,_,_ = evaluate(features, support, y_val, val_mask, sample_z, placeholders)
    print("\t\tClassifier validation accuracy: ", v_acc)
    print("\t\tClassifier validation precision:", v_pre)
    print("\t\tClassifier validation recall:", v_recall)
    print("\t\tClassifier validation f1_score:", v_f1)

    ##add the training discriminator loss
    val_d_loss.append(v_discriminator_loss)

    f1_score_val.append(v_f1)

    acc_test, pre_test, recall_test, f1_test, discriminator_loss_test, duration_test, _ = evaluate(features, support,
                                                                                                   y_test, test_mask,
                                                                                                   sample_z,
                                                                                                   placeholders)
    print("\t\tClassifier testing accuracy: ", acc_test)
    print("\t\tClassifier testing precision:", pre_test)
    print("\t\tClassifier testing recall:", recall_test)
    print("\t\tClassifier testing f1_score:", f1_test)


    saver.save(sess, './' + checkpoint_path + 'my_test_model', global_step=epoch)
    if f1_score_val[-1] >= best:
        best = f1_score_val[-1]
        best_epoch = epoch
        bad_counter = 0
        print("the best epoch is %d" % best_epoch)
    else:
        bad_counter += 1

    if bad_counter == FLAGS.early_stopping:
        break

    files = glob.glob('./' + checkpoint_path + '*.index')
    for file in files:
        epoch_nb = file.split('.index')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb < best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.meta')
    for file in files:
        epoch_nb = file.split('.meta')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb < best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.data*')
    for file in files:
        epoch_nb = file.split('.data')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb < best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.index')
    for file in files:
        epoch_nb = file.split('.index')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb > best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.meta')
    for file in files:
        epoch_nb = file.split('.meta')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb > best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.data*')
    for file in files:
        epoch_nb = file.split('.data')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb > best_epoch:
            os.remove(file)


print(""
      ""
      ""
      "Optimization Finished!")


with open(FLAGS.datasetname+'test.txt', 'w') as f:
    train_time=format(time.time() - t_total)
    f.write("The training time is: \n")
    f.write("%s" % str(train_time)+ 's'+'\n')
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
'''
'''
## select a query strategy
sample_methods =['random','entropy','margin']
#sampling_method ='random'
sess = tf.Session()
#enable to directly load pre-trained model
best_epoch = 39
#best_f1 = 0.0
#val_loss = 100

new_saver = tf.train.import_meta_graph('./'+ checkpoint_path+'saved_model/'+'my_test_model-'+str(best_epoch)+'.meta')
new_saver.restore(sess, './' + checkpoint_path+'saved_model/' + 'my_test_model-'+str(best_epoch))
# Now, let's access and create placeholders variables
graph = tf.get_default_graph()
'''
'''
print("The best epoch is %d" %best_epoch)
seed = FLAGS.seed
original_train_labeled_num = train_mask.sum()
##backup the train_active_mask, deepcopy
back_train_mask = copy.deepcopy(train_mask)
back_train_active_mask = copy.deepcopy(train_active_mask)
update_train_mask = train_mask
update_train_active_mask = train_active_mask
best_f1_dict ={}
best_train_loss_dict = {}
for method in sample_methods:
    new_saver = tf.train.import_meta_graph(
        './' + checkpoint_path + 'saved_model/' + 'my_test_model-' + str(best_epoch) + '.meta')
    new_saver.restore(sess, './' + checkpoint_path + 'saved_model/' + 'my_test_model-' + str(best_epoch))
    # Now, let's access and create placeholders variables
    graph = tf.get_default_graph()
    assert back_train_mask.sum()==original_train_labeled_num
    update_train_active_mask = copy.deepcopy(back_train_active_mask)
    best_f1 = 0.0
    train_loss = 100
    update_train_mask = copy.deepcopy(back_train_mask)
    num_examples = train_mask.sum()
    for epoch in range(FLAGS.query_epoch):
        print('learning rate {}'.format(sess.run([model.learning_rate])))

        print('\n Query strategy search for epoch {} :'.format(epoch + 1))
        acc_active_train, pre_active_train, recall_active_train, f1_active_train, discriminator_loss_active_train, duration_active_train, dis_active_data, data_embeds = evaluate(
            features, support, y_train_active, update_train_active_mask, sample_z, placeholders)
        # print(acc_active_train)
        # print(f1_active_train)
        if method == 'random':
            sampler_id, update_train_active_mask = random_sampler(FLAGS.query_sample_size, update_train_active_mask, seed)
            print(sampler_id)
        elif method == 'entropy':
            active_entropy_score = entropy_score(dis_active_data)
            sampler_id, update_train_active_mask = entropy_sampler(FLAGS.query_sample_size, update_train_active_mask, active_entropy_score)
            print(sampler_id)
        elif method == 'margin':
            sampler_id, update_train_active_mask = margin_sampler(FLAGS.query_sample_size, update_train_active_mask, dis_active_data)

        update_train_mask[sampler_id] = True
        #print(back_train_mask.sum())
        #print(back_train_mask[sampler_id[0]])

        t1e = time.time()
        num_examples = update_train_mask.sum()
        assert num_examples == (epoch + 1) * FLAGS.query_sample_size + original_train_labeled_num
        num_correct = 0
        # Construct feed dictionary for the training part
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['inputs_z']: sample_z})

        # freeze the generator part, don't touch g_opt

        _, correct, masked_t_pre, masked_t_label, discriminator_loss, generator_loss = sess.run(
            [model.d_opt, model.masked_correct,
             model.masked_pred_class, model.masked_labels, model.d_loss, model.g_loss], feed_dict=feed_dict)


        num_correct += correct
        ##add the training discriminator loss
        train_d_loss.append(discriminator_loss)
        train_g_loss.append(generator_loss)
        sess.run([model.shrink_lr])
        train_accuracy = num_correct / float(num_examples)
        y_true = np.argmax(masked_t_label, axis=1)
        y_pred = masked_t_pre
        # print("y_true's shape is"+ str(y_true.shape))
        # print("y_pred's shape is"+ str(y_pred.shape))
        t_precision, t_recall, t_f1_score, t_support = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                                               average=None)
        print("\t\tClassifier train accuracy: ", train_accuracy)
        t_accu = metrics.accuracy_score(y_true, y_pred, normalize=True)
        print("\t\tClassifier train accuracy validation:", t_accu)
        print("\t\tClassifier train precision:", t_precision[-1])
        print("\t\tClassifier train recall:", t_recall[-1])
        print("\t\tClassifier train f1_score:", t_f1_score[-1])

        # Validation
        v_acc, v_pre, v_recall, v_f1, v_discriminator_loss, duration, _,_ = evaluate(features, support, y_val, val_mask,
                                                                                   sample_z, placeholders)
        print("\t\tClassifier validation accuracy: ", v_acc)
        print("\t\tClassifier validation precision:", v_pre)
        print("\t\tClassifier validation recall:", v_recall)
        print("\t\tClassifier validation f1_score:", v_f1)
        print("\t\t Validation loss:", v_discriminator_loss)

        ##add the training discriminator loss

        f1_score_train.append(t_f1_score[-1])

        if f1_score_train[-1] >= best_f1 or train_d_loss[-1] <= train_loss:
            if f1_score_train[-1] >= best_f1:
                best_f1 = f1_score_train[-1]
            if train_d_loss[-1] <= train_loss:
                train_loss = train_d_loss[-1]
            #best_epoch = epoch

    ##get the best f1_score_val based on the validation performance
    best_f1_dict[method] = best_f1
    best_train_loss_dict[method] = train_loss

##according to the validation score choose the best sampling method
##Current the conclusion is not effective to choose a strategy
'''




train_d_loss, val_d_loss =  [], []
train_g_loss =[]
f1_score_val=[]

# Testing
sess = tf.Session()
#enable to directly load pre-trained model
best_epoch = 39
new_saver = tf.train.import_meta_graph('./'+ checkpoint_path+'saved_model/'+'my_test_model-'+str(best_epoch)+'.meta')
#new_saver = tf.train.import_meta_graph('./'+ checkpoint_path+'my_test_model-'+str(best_epoch)+'.meta')
new_saver.restore(sess, './' + checkpoint_path+'saved_model/' + 'my_test_model-'+str(best_epoch))
#new_saver.restore(sess, './' + checkpoint_path+ 'my_test_model-'+str(best_epoch))

# Now, let's access and create placeholders variables
graph = tf.get_default_graph()

discriminator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
print("discriminator_train_vars are: ")
print("////////////Debug purpose, discriminator variables/////////////")
for discriminator_train_var in discriminator_train_vars:
    print (discriminator_train_var)
print("///////////Debug purpose, discriminator variables//////////////")

generator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

print("genarator_train_vars are: ")
print("/////////////Debug purpose, generator variables////////////////")
for generator_train_var in generator_train_vars:
    print (generator_train_var)
print("/////////////Debug purpose, generator variables////////////////")

print("The best epoch is %d" %best_epoch)



acc_test, pre_test, recall_test, f1_test, discriminator_loss_test, duration_test,_,_ = evaluate(features, support, y_test, test_mask, sample_z,placeholders)
print("\t\tClassifier testing accuracy: ", acc_test)
print("\t\tClassifier testing precision:", pre_test)
print("\t\tClassifier testing recall:", recall_test)
print("\t\tClassifier testing f1_score:", f1_test)



###continue writing the test performance into a txt file
with open(FLAGS.datasetname+'gedet_test.txt', 'a') as f:
    f.write("The testing accuracy is: \n")
    f.write("%s" % str(acc_test)+ '\n')
    f.write("The testing precision is: \n")
    f.write("%s" % str(pre_test)+ '\n')
    f.write("The testing recall is: \n")
    f.write("%s" % str(recall_test)+ '\n')
    f.write("The testing f1_score is: \n" )
    f.write("%s" % str(f1_test) + '\n')

###start active learning, first implementing a random_sampling based model
##sampling a random step-size instance from
original_train_labeled_num = train_mask.sum()
# freeze the generator part, don't touch g_opt
frozen_variables = [v for v in tf.trainable_variables() if 'generator/' in v.name]
updated_variables = [v for v in tf.trainable_variables() if 'discriminator/' in v.name]
sampling_method = FLAGS.sample_method
seed = FLAGS.seed
#sess.run(model.learning_rate)
if 'dm' in FLAGS.datasetname:
    update = model.learning_rate.assign(sess.run(model.learning_rate)*pow((1), 5))
    sess.run(update)

elif 'ml' in FLAGS.datasetname:
    update = model.learning_rate.assign(sess.run(model.learning_rate)*pow((0.9), 5))
    sess.run(update)

elif 'user17' in FLAGS.datasetname:
    update = model.learning_rate.assign(sess.run(model.learning_rate)*pow((0.9), 15))
    sess.run(update)

elif 'user10' in FLAGS.datasetname:
    update = model.learning_rate.assign(sess.run(model.learning_rate)*pow((1), 5))
    sess.run(update)

elif 'species' in FLAGS.datasetname:
    update = model.learning_rate.assign(sess.run(model.learning_rate)*pow((0.6), 3))
    sess.run(update)



best = 0.0
val_loss = 100
for epoch in range(FLAGS.active_epochs):
    t1e = time.time()

    print('learning rate {}'.format(sess.run([model.learning_rate])))

    print('\nActive Learning for epoch {} :'.format(epoch + 1))
    acc_active_train, pre_active_train, recall_active_train, f1_active_train, discriminator_loss_active_train, duration_active_train, dis_active_data,data_embeds = evaluate(features, support, y_train_active, train_active_mask, sample_z,placeholders)
    #print(acc_active_train)
    #print(f1_active_train)
    print("start implementing sampler")
    if sampling_method =='random':
        '''
        sampler_id, train_active_mask =random_sampler(FLAGS.sample_size, ground_truth_train_active_mask, seed,ground_truth_train_active)
        ground_truth_train_active_mask = train_active_mask
        '''
        sampler_id, train_active_mask = random_sampler(FLAGS.sample_size, train_active_mask, seed,ground_truth_train_active)
    elif sampling_method =='entropy':
        active_entropy_score = entropy_score(dis_active_data)
        sampler_id, train_active_mask = entropy_sampler(FLAGS.sample_size, train_active_mask, active_entropy_score,ground_truth_train_active)
    elif sampling_method == 'margin':
        sampler_id, train_active_mask = margin_sampler(FLAGS.sample_size, train_active_mask, dis_active_data)
    elif sampling_method == 'kmeans':
        active_data_embeds = data_embeds[train_active_mask]
        #train_active_mask[ground_truth_train_active_mask == True] = False
        #active_data_embeds = data_embeds[train_active_mask]
        #active_data_embeds = data_embeds[ground_truth_train_active_mask]
        #print(sum(ground_truth_train_active_mask))
        print(sum(train_active_mask))
        sampler_id, train_active_mask = kmeans_sampler(FLAGS.cluster_size, FLAGS.sample_size, train_active_mask, active_data_embeds,ground_truth_train_active)

        '''
        sampler_id, train_active_mask = kmeans_sampler(FLAGS.sample_size, ground_truth_train_active_mask, active_data_embeds,
                                                       ground_truth_train_active)
        '''

    elif  sampling_method=='approximation':
        lam = FLAGS.lamda
        active_data_embeds = data_embeds[train_active_mask]
        print(sum(train_active_mask))
        sampler_id, train_active_mask = approximation_sampler(FLAGS.cluster_size, FLAGS.sample_size, train_active_mask,
                                                       active_data_embeds, ground_truth_train_active, lam)

    elif sampling_method =='un_approximation':
        lam = FLAGS.lamda
        active_data_embeds = data_embeds[train_active_mask]
        print(sum(train_active_mask))
        sampler_id, train_active_mask = un_approximation_sampler(FLAGS.cluster_size, FLAGS.sample_size, train_active_mask,
                                                              active_data_embeds, ground_truth_train_active, lam)

    elif sampling_method == 'approximation_memorization':
        lam = FLAGS.lamda
        active_data_embeds = data_embeds[train_active_mask]
        print(sum(train_active_mask))
        sampler_id, train_active_mask = approximation_memo_sampler(FLAGS.cluster_size, FLAGS.sample_size,
                                                                 train_active_mask,
                                                                 active_data_embeds, ground_truth_train_active, lam)






    train_mask[sampler_id]= True

    num_examples = train_mask.sum()
    assert num_examples == (epoch+1)*FLAGS.sample_size + original_train_labeled_num
    num_correct = 0
    # Construct feed dictionary for the training part
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['inputs_z']: sample_z})


    #freeze the generator part, don't touch g_opt

    _, correct, masked_t_pre, masked_t_label, discriminator_loss, generator_loss = sess.run(
        [model.d_opt, model.masked_correct,
         model.masked_pred_class, model.masked_labels, model.d_loss, model.g_loss], feed_dict=feed_dict)
    tmp_frozen_variables_np = sess.run(frozen_variables)
    for i in range(len(frozen_variables)):
        assert (np.allclose(tmp_frozen_variables_np[i],sess.run(frozen_variables[i])))

    '''
    tmp_updated_variables_np = sess.run(updated_variables)
    for i in range(len(updated_variables)):
        print(sess.run(updated_variables[i]))
    '''
        #print (np.allclose(tmp_updated_variables_np[i], sess.run(updated_variables[i])))

    num_correct += correct
    ##add the training discriminator loss
    train_d_loss.append(discriminator_loss)
    train_g_loss.append(generator_loss)
    sess.run([model.shrink_lr])
    train_accuracy = num_correct / float(num_examples)
    y_true = np.argmax(masked_t_label, axis=1)
    y_pred = masked_t_pre
    # print("y_true's shape is"+ str(y_true.shape))
    # print("y_pred's shape is"+ str(y_pred.shape))
    t_precision, t_recall, t_f1_score, t_support = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    print("\t\tClassifier train accuracy: ", train_accuracy)
    t_accu = metrics.accuracy_score(y_true, y_pred, normalize=True)
    print("\t\tClassifier train accuracy validation:", t_accu)
    print("\t\tClassifier train precision:", t_precision[-1])
    print("\t\tClassifier train recall:", t_recall[-1])
    print("\t\tClassifier train f1_score:", t_f1_score[-1])
    epoch_stop = timeit.default_timer()
    print("\t\tActive Learning training time:", epoch_stop-t1e)

    # Validation
    v_acc, v_pre, v_recall, v_f1, v_discriminator_loss, duration,_,_ = evaluate(features, support, y_val, val_mask,
                                                                            sample_z, placeholders)
    print("\t\tClassifier validation accuracy: ", v_acc)
    print("\t\tClassifier validation precision:", v_pre)
    print("\t\tClassifier validation recall:", v_recall)
    print("\t\tClassifier validation f1_score:", v_f1)
    print("\t\t Validation loss:", v_discriminator_loss)

    ##add the training discriminator loss
    val_d_loss.append(v_discriminator_loss)

    f1_score_val.append(v_f1)

    saver.save(sess, './' + checkpoint_path +sampling_method+ 'my_test_model', global_step=epoch)

    best_epoch = epoch

    '''
    if f1_score_val[-1] >= best or val_d_loss[-1]<=val_loss:
        if f1_score_val[-1] >= best:
            best = f1_score_val[-1]
        if val_d_loss[-1]<= val_loss:
            val_loss = val_d_loss[-1]
        best_epoch = epoch
        bad_counter = 0
        print("the best epoch is %d" % best_epoch)
    else:
        bad_counter += 1

    if bad_counter == FLAGS.early_stopping:
        break
    '''

    files = glob.glob('./' + checkpoint_path + '*.index')
    for file in files:
        epoch_nb = file.split('.index')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb < best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.meta')
    for file in files:
        epoch_nb = file.split('.meta')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb < best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.data*')
    for file in files:
        epoch_nb = file.split('.data')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb < best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.index')
    for file in files:
        epoch_nb = file.split('.index')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb > best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.meta')
    for file in files:
        epoch_nb = file.split('.meta')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb > best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.data*')
    for file in files:
        epoch_nb = file.split('.data')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb > best_epoch:
            os.remove(file)


    acc_test, pre_test, recall_test, f1_test, discriminator_loss_test, duration_test, _,_ = evaluate(features, support,
                                                                                                   y_test, test_mask,
                                                                                                   sample_z,
                                                                                                   placeholders)
    print("\t\tClassifier testing accuracy: ", acc_test)
    print("\t\tClassifier testing precision:", pre_test)
    print("\t\tClassifier testing recall:", recall_test)
    print("\t\tClassifier testing f1_score:", f1_test)


new_saver = tf.train.import_meta_graph('./'+ checkpoint_path+sampling_method+'my_test_model-'+str(best_epoch)+'.meta')
new_saver.restore(sess, './' + checkpoint_path+sampling_method + 'my_test_model-'+str(best_epoch))
acc_test, pre_test, recall_test, f1_test, discriminator_loss_test, duration_test,_,_ = evaluate(features, support, y_test, test_mask, sample_z,placeholders)
print("\t\tClassifier testing accuracy: ", acc_test)
print("\t\tClassifier testing precision:", pre_test)
print("\t\tClassifier testing recall:", recall_test)
print("\t\tClassifier testing f1_score:", f1_test)

with open(FLAGS.datasetname+'gedet++_test.txt', 'a') as f:
    f.write("The testing accuracy is: \n")
    f.write("%s" % str(acc_test)+ '\n')
    f.write("The testing precision is: \n")
    f.write("%s" % str(pre_test)+ '\n')
    f.write("The testing recall is: \n")
    f.write("%s" % str(recall_test)+ '\n')
    f.write("The testing f1_score is: \n" )
    f.write("%s" % str(f1_test) + '\n')

stop = timeit.default_timer()

print('The total time:', stop-start)

copyfile('logfile.log',os.path.join(run_dir,+run_id+'logfile.log'))
