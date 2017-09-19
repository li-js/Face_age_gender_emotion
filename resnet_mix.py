from collections import namedtuple

import tensorflow as tf
import numpy as np

import sys
import utils


HParams = namedtuple('HParams',
                    'batch_size, batch_indices, num_gpus, '
                    'num_classes_age, num_classes_gender, num_classes_smile, num_classes_glass, input_size, '
                    'weight_decay, momentum, finetune')

class ResNet(object):
    def __init__(self, hp, global_step, name=None, reuse_weights=False):

        self.lr = tf.placeholder(tf.float32, name="lr")
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.gpu0_images = tf.placeholder(tf.float32, name='gpu0_images', shape=(hp.batch_size, hp.input_size, hp.input_size, 3))
        self.gpu0_labels_age = tf.placeholder(tf.int32, name='gpu0_labels_age', shape=(hp.batch_indices['t1_batch']))
        self.gpu0_labels_gender = tf.placeholder(tf.int32, name='gpu0_labels_gender', shape=(hp.batch_indices['t2_batch']))
        self.gpu0_labels_smile = tf.placeholder(tf.int32, name='gpu0_labels_smile', shape=(hp.batch_indices['t2_batch']))
        self.gpu0_labels_glass = tf.placeholder(tf.int32, name='gpu0_labels_glass', shape=(hp.batch_indices['t2_batch']))

        if hp.num_gpus == 2:
            self.gpu1_images = tf.placeholder(tf.float32, name='gpu1_images', shape=(hp.batch_size, hp.input_size, hp.input_size, 3))
            self.gpu1_labels_age = tf.placeholder(tf.int32, name='gpu0_labels_age', shape=(hp.batch_indices['t1_batch']))
            self.gpu1_labels_gender = tf.placeholder(tf.int32, name='gpu1_labels_gender', shape=(hp.batch_indices['t2_batch']))
            self.gpu1_labels_smile = tf.placeholder(tf.int32, name='gpu1_labels_smile', shape=(hp.batch_indices['t2_batch']))
            self.gpu1_labels_glass = tf.placeholder(tf.int32, name='gpu1_labels_glass', shape=(hp.batch_indices['t2_batch']))

            self._images = [self.gpu0_images, self.gpu1_images]
            self._labels_age = [self.gpu0_labels_age, self.gpu1_labels_age]
            self._labels_gender = [self.gpu0_labels_gender, self.gpu1_labels_gender]
            self._labels_smile = [self.gpu0_labels_smile, self.gpu1_labels_smile]
            self._labels_glass = [self.gpu0_labels_glass, self.gpu1_labels_glass]
        elif hp.num_gpus == 1:
            self._images = [self.gpu0_images]
            self._labels_age = [self.gpu0_labels_age]
            self._labels_gender = [self.gpu0_labels_gender]
            self._labels_smile = [self.gpu0_labels_smile]
            self._labels_glass = [self.gpu0_labels_glass]
        else:
            assert(0)

        self._hp = hp # Hyperparameters
        self._global_step = global_step
        self._name = name
        self._reuse_weights = reuse_weights
        self._counted_scope = []
        self._flops = 0
        self._weights = 0


    def build_tower(self, images, labels_age, labels_gender, labels_smile, labels_glass):
        print('Building model')
        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(images, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
        x = self._residual_block(x, name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
        x = self._residual_block(x, name='conv5_2')

        common_feat = tf.reduce_mean(x, [1, 2])
        if self._hp.batch_indices['use_split']:
            t1_out_ = common_feat[self._hp.batch_indices['t1'][0]:self._hp.batch_indices['t1'][1]]
            t2_out_ = common_feat[self._hp.batch_indices['t2'][0]:self._hp.batch_indices['t2'][1]]
        else:
            t1_out_ = common_feat
            t2_out_ = common_feat

        # Logit
        with tf.variable_scope('logits_age') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_age = self._fc(t1_out_, self._hp.num_classes_age)

        with tf.variable_scope('logits_gender') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_gender = self._fc(t2_out_, self._hp.num_classes_gender)

        with tf.variable_scope('logits_smile') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_smile  = self._fc(t2_out_, self._hp.num_classes_smile)

        with tf.variable_scope('logits_glass') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_glass  = self._fc(t2_out_, self._hp.num_classes_glass)

        logits_age = x_age
        logits_gender = x_gender
        logits_smile = x_smile
        logits_glass = x_glass


        # Loss & acc, Probs & preds
        #probs = tf.nn.softmax(x)
        loss_age = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_age, labels=labels_age))
        preds_age = tf.to_int32(tf.argmax(logits_age, 1))
        correct_age = tf.abs(preds_age - labels_age)
        mae_age = tf.reduce_mean(tf.cast(correct_age, tf.float32))

        loss_gender = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_gender, labels=labels_gender))
        preds_gender = tf.to_int32(tf.argmax(logits_gender, 1))
        correct_gender = tf.equal(preds_gender, labels_gender)
        acc_gender = tf.reduce_mean(tf.cast(correct_gender, tf.float32))

        loss_smile = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_smile, labels=labels_smile))
        preds_smile = tf.to_int32(tf.argmax(logits_smile, 1))
        correct_smile = tf.equal(preds_smile, labels_smile)
        acc_smile = tf.reduce_mean(tf.cast(correct_smile, tf.float32))

        loss_glass = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_glass, labels=labels_glass))
        preds_glass = tf.to_int32(tf.argmax(logits_glass, 1))
        correct_glass = tf.equal(preds_glass, labels_glass)
        acc_glass = tf.reduce_mean(tf.cast(correct_glass, tf.float32))

        return {'age': {'logits': logits_age, 'preds':preds_age, 'loss': loss_age, 'mae': mae_age}, 
                'gender': {'logits': logits_gender, 'preds':preds_gender, 'loss': loss_gender, 'acc': acc_gender}, 
                'smile': {'logits': logits_smile, 'preds':preds_smile, 'loss': loss_smile, 'acc': acc_smile}, 
                'glass': {'logits': logits_glass, 'preds':preds_glass, 'loss': loss_glass, 'acc': acc_glass} }


    def build_model(self):
        # Split images and labels into (num_gpus) groups
        # images = tf.split(self._images, num_or_size_splits=self._hp.num_gpus, axis=0)
        # labels = tf.split(self._labels, num_or_size_splits=self._hp.num_gpus, axis=0)

        # Build towers for each GPU
        self._logits_age_list = []
        self._preds_age_list = []
        self._loss_age_list = []
        self._mae_age_list = []


        self._logits_gender_list = []
        self._preds_gender_list = []
        self._loss_gender_list = []
        self._acc_gender_list = []

        self._logits_smile_list = []
        self._preds_smile_list = []
        self._loss_smile_list = []
        self._acc_smile_list = []

        self._logits_glass_list = []
        self._preds_glass_list = []
        self._loss_glass_list = []
        self._acc_glass_list = []        

        for i in range(self._hp.num_gpus):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i) as scope:
                    print('Build a tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()

                    all_output_tensors = self.build_tower(self._images[i],  self._labels_age[i], 
                                self._labels_gender[i], self._labels_smile[i], self._labels_glass[i])

                    self._logits_age_list.append(all_output_tensors['age']['logits'])
                    self._preds_age_list.append(all_output_tensors['age']['preds'])
                    self._loss_age_list.append(all_output_tensors['age']['loss'])
                    self._mae_age_list.append(all_output_tensors['age']['mae'])


                    self._logits_gender_list.append(all_output_tensors['gender']['logits'])
                    self._preds_gender_list.append(all_output_tensors['gender']['preds'])
                    self._loss_gender_list.append(all_output_tensors['gender']['loss'])
                    self._acc_gender_list.append(all_output_tensors['gender']['acc'])

                    self._logits_smile_list.append(all_output_tensors['smile']['logits'])
                    self._preds_smile_list.append(all_output_tensors['smile']['preds'])
                    self._loss_smile_list.append(all_output_tensors['smile']['loss'])
                    self._acc_smile_list.append(all_output_tensors['smile']['acc'])

                    self._logits_glass_list.append(all_output_tensors['glass']['logits'])
                    self._preds_glass_list.append(all_output_tensors['glass']['preds'])
                    self._loss_glass_list.append(all_output_tensors['glass']['loss'])
                    self._acc_glass_list.append(all_output_tensors['glass']['acc'])

        # Merge losses, accuracies of all GPUs
        with tf.device('/CPU:0'):
            self.logits_age = tf.concat(self._logits_age_list, axis=0, name="logits_age")
            self.preds_age = tf.concat(self._preds_age_list, axis=0, name="predictions_age")
            self.loss_age = tf.reduce_mean(self._loss_age_list, name="cross_entropy_age")
            self.mae_age = tf.reduce_mean(self._mae_age_list, name="mae_age")
            tf.summary.scalar((self._name+"/" if self._name else "") + "cross_entropy_age", self.loss_age)            
            tf.summary.scalar((self._name+"/" if self._name else "") + "mae_age", self.mae_age)


            self.logits_gender = tf.concat(self._logits_gender_list, axis=0, name="logits_gender")
            self.preds_gender = tf.concat(self._preds_gender_list, axis=0, name="predictions_gender")
            self.loss_gender = tf.reduce_mean(self._loss_gender_list, name="cross_entropy_gender")
            self.acc_gender = tf.reduce_mean(self._acc_gender_list, name="accuracy_gender")
            tf.summary.scalar((self._name+"/" if self._name else "") + "cross_entropy_gender", self.loss_gender)            
            tf.summary.scalar((self._name+"/" if self._name else "") + "accuracy_gender", self.acc_gender)

            self.logits_smile = tf.concat(self._logits_smile_list, axis=0, name="logits_smile")
            self.preds_smile = tf.concat(self._preds_smile_list, axis=0, name="predictions_smile")
            self.loss_smile = tf.reduce_mean(self._loss_smile_list, name="cross_entropy_smile")
            self.acc_smile = tf.reduce_mean(self._acc_smile_list, name="accuracy_smile")
            tf.summary.scalar((self._name+"/" if self._name else "") + "cross_entropy_smile", self.loss_smile)            
            tf.summary.scalar((self._name+"/" if self._name else "") + "accuracy_smile", self.acc_smile)

            self.logits_glass = tf.concat(self._logits_glass_list, axis=0, name="logits_glass")
            self.preds_glass = tf.concat(self._preds_glass_list, axis=0, name="predictions_glass")
            self.loss_glass = tf.reduce_mean(self._loss_glass_list, name="cross_entropy_glass")
            self.acc_glass = tf.reduce_mean(self._acc_glass_list, name="accuracy_glass")
            tf.summary.scalar((self._name+"/" if self._name else "") + "cross_entropy_glass", self.loss_glass)            
            tf.summary.scalar((self._name+"/" if self._name else "") + "accuracy_glass", self.acc_glass)


    def build_train_op(self):
        # Learning rate
        tf.summary.scalar((self._name+"/" if self._name else "") + 'learing_rate', self.lr)

        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        self._grads_and_vars_list = []

        # Computer gradients for each GPU
        for i in range(self._hp.num_gpus):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i) as scope:
                    print('Compute gradients of tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()

                    # Add l2 loss
                    costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
                    l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
                    total_loss = l2_loss + self._loss_age_list[i] + \
                    0.333*(self._loss_gender_list[i] + self._loss_smile_list[i] + self._loss_glass_list[i])

                    # Compute gradients of total loss
                    grads_and_vars = opt.compute_gradients(total_loss, tf.trainable_variables())

                    # Append gradients and vars
                    self._grads_and_vars_list.append(grads_and_vars)

        # Merge gradients
        print('Average gradients')
        with tf.device('/CPU:0'):
            grads_and_vars = self._average_gradients(self._grads_and_vars_list)

            # Finetuning
            if self._hp.finetune:
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if "unit3" in var.op.name or \
                        "unit_last" in var.op.name or \
                        "/q" in var.op.name or \
                        "logits" in var.op.name:
                        print('\tScale up learning rate of % s by 10.0' % var.op.name)
                        grad = 10.0 * grad
                        grads_and_vars[idx] = (grad,var)

            # Apply gradient
            apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

            # Batch normalization moving average update
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group(*(update_ops+[apply_grad_op]))


    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # If no gradient for a variable, exclude it from output
            if grad_and_vars[0][0] is None:
                continue

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads


    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_train, self._global_step, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)
