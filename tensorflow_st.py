import tensorflow as tf


def gumbel_ST(logits, temp=1.0, hard=False):
    eps = 1e-8
    gumbel_noise = -tf.log(-tf.log(tf.random_uniform(tf.shape(logits)) + eps)
                           + eps)

    y = tf.nn.softmax((logits + gumbel_noise) / temp)

    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)

        y = tf.stop_gradient(y_hard - y) + y

    return y


if __name__ == '__main__':
    a = tf.random_uniform((4, 10))
    c = tf.random_uniform((4, 10))
    b = tf.random_uniform((10, 5))

    sample = gumbel_ST(a * c, hard=True)

    out = tf.reduce_sum(tf.matmul(sample, b))

    with tf.Session() as sess:
        grad = tf.gradients(out, c)[0].eval(session=sess)

    print(grad)
