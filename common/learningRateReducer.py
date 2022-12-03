import tensorflow as tf

class LearningRateReducer(tf.keras.callbacks.Callback):

  def __init__(self, downscale_factor):
    self.downscale_factor = downscale_factor

  def on_epoch_end(self, epoch):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * self.downscale_factor
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}\n".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)