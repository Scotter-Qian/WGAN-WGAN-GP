import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time


#-------------------------制作tf.record文件-----------------------------
#定义函数搜索要训练的数据路径
def search_file(path):
    file = list()
    for name in os.listdir(path):
        name_path = os.path.join(path, name)
        if os.path.isfile(name_path):
            if name_path.endswith(".xy"):
                file.append(name_path)
    return np.array(file)


# write images and label into tfrecord
def encode_to_tfrecords(path):
    #train时将"valid.tfrecord"改为"train.tfrecord"
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    data_path_file = search_file(path)
    for file_path in data_path_file:
        file = open(file_path, "rb")
        data = pickle.load(file)

        image = (data["x"])
        image = np.reshape(image, (288, 384))
        label = (data["y"])
        #[2]表示类别数为2
        label = np.reshape(label, 1)

        image_raw = image.tobytes()
        #label_raw = label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
        writer.write(example.SerializeToString())
    writer.close()
    print("tfrecord_file is done")


def read_example(filename, batch_size):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    _, serialized_example = reader.read(filename_queue)
    min_queue_examples = 50
    batch = tf.train.shuffle_batch([serialized_example], batch_size=batch_size,
                                   capacity=min_queue_examples + 100 * batch_size, min_after_dequeue=min_queue_examples,
                                   num_threads=2)
    parsed_example = tf.parse_example(batch, features={'image': tf.FixedLenFeature([], tf.string),
                                                       'label': tf.FixedLenFeature([], tf.int64)})
    image_raw = tf.decode_raw(parsed_example['image'], tf.uint8)
    #IMAGE_HEIGHT为288，IMAGE_WIDTH为384, IMAGE_DEPTH为1
    image = tf.cast(tf.reshape(image_raw, [batch_size, 288, 384, 1]), tf.float32)
    image = image/255.0
    label_raw = tf.cast(parsed_example['label'], tf.int32)
    label = tf.reshape(label_raw, [batch_size*1])
    #depth=num_classes,这里是二分类，所以num_classes=2
    label = tf.one_hot(label, depth=2)
    return image, label

#-------------------------------保存图片---------------------------------------

def save_image(images, size):
    image = (images + 1.0)*(255.99/2)
    h, w = image.shape[1], image.shape[2]
    print(h, w)
    merge_image = np.zeros((h*size[0], w*size[1]))
    for idx in range(image.shape[0]):
        i = idx % size[1]
        j = idx // size[1]
        merge_image[j*h:j*h+h, i*w:i*w+w] = image[idx]
    return merge_image

#-------------------------------------画损失函数-------------------------------

def draw_picture(data_1, data_2):
    train_loss_array_1 = np.array(data_1)
    train_loss_array_2 = np.array(data_2)
    plt.figure(figsize=[8,6])
    plt.plot(train_loss_array_1[:,0], train_loss_array_1[:,1], 'r', label='Loss_d', linewidth=2.0)
    plt.plot(train_loss_array_2[:, 0], train_loss_array_2[:, 1], 'b', label='Loss_g', linewidth=2.0)
    plt.title('train_loss(vgg)', fontsize=24)
    plt.xlabel('step', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.legend()
    plt.show()

#-------------------------------------构建网络----------------------------------


def conv_layer(input, kernel_size, out_channels, strides, name, batch_normalize, training=True):
    in_channels = input.get_shape()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable(name="w", shape=[kernel_size, kernel_size, in_channels, out_channels],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
        b = tf.get_variable(name="b", shape=[out_channels],
                            dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=training)
        x = tf.nn.conv2d(input, w, strides, padding="SAME", name="conv")
        x = tf.nn.bias_add(x, b, name="add")
        #如果有BN层，则BN层应放在tf.layers.conv2d和activation之间
        if batch_normalize:
            x = tf.layers.batch_normalization(x, training=training,  momentum=0.99, epsilon=1e-5, scale=True)
        return x


def deconv_layer(inputs, kernel_size, out_channels, output_shape, strides, name, batch_normalize, training=True):
    in_channels = inputs.get_shape()[-1]
    with tf.variable_scope(name):
        weights = tf.get_variable('w', shape=[kernel_size, kernel_size, out_channels, in_channels],
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=training)
        #print(weights.name)
        biases = tf.get_variable('b', shape=[out_channels],
                                 dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=training)
        deconv = tf.nn.conv2d_transpose(inputs, weights, output_shape=output_shape, strides=strides, padding='SAME', name="deconv")
        deconv  = tf.add(deconv, biases, name="add")
        if batch_normalize:
            deconv = tf.layers.batch_normalization(deconv, training=training, momentum=0.99, epsilon=0.1e-5, scale=None)

        return deconv


def fc_layer(input, in_size, out_size, name, batch_normalize, training=True):
    with tf.variable_scope(name):
        weights = tf.get_variable('w', shape=[in_size, out_size],
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
        biases = tf.get_variable('b', shape=[out_size],
                                initializer=tf.constant_initializer(0.0), trainable=training)
        fc =  tf.matmul(input, weights) + biases
        if batch_normalize:
            fc = tf.layers.batch_normalization(fc, training=training,  momentum=0.99, epsilon=0.1e-5, scale=None)

        return fc


def generator(inputs_z, batch_size):
    in_channel = inputs_z.get_shape()[-1]
    in_height = int(IMAGE_HEIGHT/16)
    in_width = int(IMAGE_WIDTH/16)
    z_t = tf.reshape(fc_layer(inputs_z, in_channel, in_height*in_width*512, name="fc_g", batch_normalize=True),
                         [batch_size, in_height, in_width, 512])
    layer_1 =tf.nn.relu(z_t, name="relu_1")
    #print(layer_1.name)
    deconv_1 = deconv_layer(inputs=layer_1, kernel_size=3, out_channels=256, output_shape=[batch_size, in_height*2,
                            in_width*2, 256], strides=[1, 2, 2, 1], batch_normalize=True, name="deconv_1")
    layer_2 = tf.nn.relu(deconv_1, name="relu_2")
    deconv_2 = deconv_layer(inputs=layer_2, kernel_size=3, out_channels=128, output_shape=[batch_size, in_height*4 ,
                            in_width*4, 128], strides=[1, 2, 2, 1], batch_normalize=True, name="deconv_2")
    layer_3 = tf.nn.relu(deconv_2, name="relu_3")
    deconv_3 = deconv_layer(inputs=layer_3, kernel_size=5, out_channels=64, output_shape=[batch_size, in_height*8,
                            in_width*8, 64], strides=[1, 2, 2, 1], batch_normalize=True, name="deconv_3")
    layer_4 = tf.nn.relu(deconv_3, name="relu_4")
    deconv_4 = deconv_layer(inputs=layer_4, kernel_size=5, out_channels=1, output_shape=[batch_size, in_height*16,
                            in_width*16, 1], strides=[1, 2, 2, 1], batch_normalize=True, name="deconv_4")
    # 加约束就用sigmoid函数归一化到0-1之间，不加约束就用tanh函数
    logits = tf.nn.sigmoid(deconv_4, name='generate_image')

    return logits


def discriminator(inputs):
    h1_1 = conv_layer(input=inputs, kernel_size=3, out_channels=64, strides=[1, 2, 2, 1], name="conv_1",
                      batch_normalize=False)
    h1 = tf.nn.leaky_relu(h1_1, name='lrelu_1')
    h2_1 = conv_layer(input=h1, kernel_size=3, out_channels=128, strides=[1, 2, 2, 1], name="conv_2",
                      batch_normalize=True)
    h2 = tf.nn.leaky_relu(h2_1, name='lrelu_2')
    h3_1 = conv_layer(input=h2, kernel_size=3, out_channels=256, strides=[1, 2, 2, 1], name="conv_3",
                      batch_normalize=True)
    h3 = tf.nn.leaky_relu(h3_1, name='lrelu_3')
    h4_1 = conv_layer(input=h3, kernel_size=3, out_channels=512, strides=[1, 2, 2, 1], name="conv_4",
                      batch_normalize=True)
    h4 = tf.nn.leaky_relu(h4_1, name='lrelu_4')
    shape = h4.get_shape()
    # caculate the features of each images: height*width*channels
    num_features = shape[1:4].num_elements()
    input = tf.reshape(h4, [-1, num_features])
    h5 = fc_layer(input=input, in_size=num_features, out_size=2, name="fc_d", batch_normalize=False)
    #print(h4.name)
    return  h5



def train():
    #从train.tfrecords中读取数据,利用cpu读取
    with tf.device("/cpu:0"):
        with tf.name_scope("train_batch"):
            ima, lab = read_example(filename, BATCH_SIZE)
    #占位符
    image = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], name="image_placehoder")
    z = tf.placeholder(dtype=tf.float32, shape=[None, NOISE_DIM], name="noise_placeholder")
    with tf.variable_scope('gen') as scope:
        with tf.name_scope("train"):
            fake_image = generator(z, BATCH_SIZE)
        scope.reuse_variables()
        with tf.name_scope("eval"):
            fake_image_eval = generator(z, BATCH_SIZE)
    with tf.variable_scope("dis") as scope:
        with tf.name_scope("real"):
            real_logits = discriminator(image)
        scope.reuse_variables()
        with tf.name_scope("fake"):
            fake_logits = discriminator(fake_image)
    with tf.name_scope("wgan_loss"):
        gen_loss = -tf.reduce_mean(fake_logits)
        dis_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
    with tf.name_scope("gan-gp_optimizer"):
        gan_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        dis_vars = [var for var in gan_vars if "dis" in var.name]
        gen_vars = [var for var in gan_vars if "gen" in var.name]
        d_optim = tf.train.RMSPropOptimizer(learning_rate=5e-5, name="dis_rmsprop").minimize(dis_loss, var_list=dis_vars)
        g_optim = tf.train.RMSPropOptimizer(learning_rate=5e-5, name="gen_rmsprop").minimize(gen_loss, var_list=gen_vars)
    with tf.name_scope("dis_clip"):
        dis_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in dis_vars]

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    Loss_d = list()
    Loss_g = list()
    start_time = time.time()
    is_train = True
    if is_train:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(num_epoches):
            for j in range(dis_iters):
                batch_image = sess.run(ima)
                batch_z = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, NOISE_DIM]).astype(np.float32)
                dis_loss_data, _d, _ = sess.run([dis_loss, d_optim, dis_clip], feed_dict={z:batch_z, image:batch_image})
            batch_z = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, NOISE_DIM]).astype(np.float32)
            gen_loss_data, _g = sess.run([gen_loss, g_optim], feed_dict={z:batch_z})
            if (i+1) % 10 == 0:
                Loss_d.append([i + 1, dis_loss_data])
                Loss_g.append([i + 1, gen_loss_data])
                print("step: {}...".format(i+1),
                      "Discriminator Loss: {:.4f}...".format(dis_loss_data),
                      "Generator Loss: {:.4f}...".format(gen_loss_data),
                      "Average time: {:.2f}...".format((time.time()-start_time)/(i+1)))
            if (i+1) % save_rate == 0:
                if not os.path.exists('generator_images(wgan)'):
                    os.mkdir(r'.\generator_images(wgan)')
                batch_noise = np.random.uniform(-1.0, 1.0, size=([BATCH_SIZE, NOISE_DIM])).astype(np.float32)
                samples_image = sess.run(fake_image, feed_dict={z:batch_noise})
                # tanh的作用将数据变到了（-1，1），利用（samples_image+1.0）/2.0的作用就是使数据大于0
                samples_image = np.reshape(samples_image*255, (ROWS, COLS, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
                #print(samples_images)
                samples_image = np.concatenate(np.concatenate(samples_image, 1), 1)
                cv2.imwrite(r'D:\gan\generator_images(wgan)\%5d.jpg'%(i+1), samples_image)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                saver.save(sess, os.path.join(model_path, 'gan'), global_step=i + 1)
        draw_picture(Loss_d, Loss_g)
        coord.request_stop()
        coord.join(threads)
        sess.close()
    else:
        #(需要更改下文件地址)
        model = tf.train.latest_checkpoint('D:\\dcgan-project\\folder_for_dcgan')
        saver.restore(sess, model)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        batch_z = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, NOISE_DIM]).astype(np.float32)
        real_images = sess.run(image)

        if not os.path.exists('generator_images(WGAN)'):
            os.mkdir(r'.\generator_images(WGAN)')
        samples_image = sess.run(fake_image, feed_dict={image:real_images, z:batch_z})
        samples_image = np.reshape((samples_image + 1.0) * 255.0/ 2.0, (ROWS, COLS, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
        samples_image = np.concatenate(np.concatenate(samples_image, 1), 1)
        cv2.imwrite(r".\genrator_images(WGAN)\new_image.jpg", samples_image)
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__=="__main__":
    IMAGE_HEIGHT = 288
    IMAGE_WIDTH = 384
    IMAGE_DEPTH = 1
    num_epoches = 100000
    LAMBDA = 10
    dis_iters = 5
    model_path = r"D:\gan\train_model(wgan)"
    #path = r"D:\dcgan-project\data\train"
    #encode_to_tfrecords(path)
    filename = r".\train.tfrecords"
    save_rate = 1000
    NOISE_DIM = 100
    ROWS = 4
    COLS = 6
    BATCH_SIZE = ROWS*COLS
    train()
"""
No log in the loss. The output of D is no longer a probility, hence we do not apply sigmoid at the output of D
Clip the weight of D(0.01)
Train D more than G (5:1)
Use RMSProp instead of ADAM
Lower learning rate(0.00005)

"""