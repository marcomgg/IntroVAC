import tensorflow as tf
import functools


loggers = []
batch_log_freq = 1000
epoch_log_freq = 1


def attach_loggers(loggers_constructors, logdir, **kwargs):
    for construcor in loggers_constructors:
        loggers.append(construcor(logdir, **kwargs))


def train_log(_func=None, *, eval=False):
    def decorator_log(func):
        @functools.wraps(func)
        def wrapper_log(*args, **kwargs):
            out = func(*args, **kwargs)
            for l in loggers:
                l.on_train_end(*out)
            return out
        return wrapper_log

    if _func is None:
        return decorator_log
    else:
        return decorator_log(_func)


def batch_log(_func=None, *, eval=False):
    def decorator_log(func):
        @functools.wraps(func)
        def wrapper_log(*args, **kwargs):
            out = func(*args, **kwargs)
            step = out[0]
            if step % batch_log_freq == 0:
                for l in loggers:
                    l.on_batch_end(*out)
                return out
        return wrapper_log

    if _func is None:
        return decorator_log
    else:
        return decorator_log(_func)


def epoch_log(_func=None, *, eval=False):
    def decorator_log(func):
        @functools.wraps(func)
        def wrapper_log(*args, **kwargs):
            out = func(*args, **kwargs)
            step = out[0]
            if step % epoch_log_freq == 0:
                for l in loggers:
                    l.on_epoch_end(*out)
                return out
        return wrapper_log

    if _func is None:
        return decorator_log
    else:
        return decorator_log(_func)


class Logger(object):
    def on_epoch_end(self, step, scalars, vectors, images, *args):
        pass

    def on_batch_end(self, step, scalars, vectors, images, *args):
        pass

    def on_train_end(self, step, scalars, vectors, images, *args):
        pass


class TfLogger(Logger):
    def __init__(self, logdir, **kwargs):
        self.manager = TfManager(logdir)

    def on_epoch_end(self, step, scalars, vectors, images, *args):
        self.manager.log(step, scalars, vectors, images)

    def on_batch_end(self, step, scalars, vector, images, *args):
        self.manager.log(step, scalars, vector, images)


class TfManager(object):

    def __init__(self, logdir):
        """Create a summary writer logging to logdir."""
        self.writer = tf.summary.create_file_writer(logdir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""

        with tf.device("/cpu:0"):
            with self.writer.as_default():
                tf.summary.scalar(tag, value, step=step)

    def image_summary(self, tag, images, step):
        """Log tensor of images"""
        with tf.device("/cpu:0"):
            with self.writer.as_default():
                tf.summary.image(tag, images, step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        with tf.device("/cpu:0"):
            with self.writer.as_default():
                tf.summary.histogram(tag, tf.convert_to_tensor(values), step=step, buckets=bins)

    def log_vector_dictionary(self, dictionary, step):
        for k, v in dictionary.items():
            self.histo_summary(k, v, step)

    def log_scalar_dictionary(self, dictionary, step):
        for k, v in dictionary.items():
            self.scalar_summary(k, v, step)

    def log_image_dictionary(self, dictionary, step):
        for k, v in dictionary.items():
            self.image_summary(k, v, step)

    def log(self, step, scalars, vectors, images):
        self.log_image_dictionary(images, step)
        self.log_scalar_dictionary(scalars, step)
        self.log_vector_dictionary(vectors, step)
