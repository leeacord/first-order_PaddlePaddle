from paddle.fluid import Variable
from paddle.fluid.dygraph import learning_rate_scheduler
class _LearningRateEpochDecay(learning_rate_scheduler.LearningRateDecay):
    """
    :api_attr: imperative
    Base class of learning rate decay, which is updated each epoch.

    Define the common interface of an _LearningRateEpochDecay.
    User should not use this class directly,
    but need to use one of it's implementation. And invoke method: `epoch()` each epoch.
    """

    def __init__(self, learning_rate, dtype=None):
        if not isinstance(learning_rate, (float, int)):
            raise TypeError(
                "The type of 'learning_rate' must be 'float, int', but received %s."
                % type(learning_rate))
        if learning_rate < 0:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))

        self.base_lr = float(learning_rate)

        self.step_num = -1
        self.dtype = dtype
        if dtype is None:
            self.dtype = "float32"
        self.learning_rate = self.create_lr_var(self.base_lr)

        self.epoch()

    # For those subclass who overload _LearningRateEpochDecay, "self.epoch_num/learning_rate" will be stored by default.
    # you can change it for your subclass.
    ## In 1.8.2post97, "self.step_num" will be stored by default instead of "self.epoch_num"
    def _state_keys(self):
        self.keys = ['step_num', 'learning_rate']

    def __call__(self):
        """
        Return last computed learning rate on current epoch.
        """
        if not isinstance(self.learning_rate, Variable):
            self.learning_rate = self.create_lr_var(self.learning_rate)
        return self.learning_rate

    def epoch(self, epoch=None):
        """
        compueted learning_rate and update it when invoked.
        """
        if epoch is None:
            self.step_num += 1
        else:
            self.step_num = epoch

        self.learning_rate = self.get_lr()

    def get_lr(self):
        raise NotImplementedError


class MultiStepDecay(_LearningRateEpochDecay):
    """
    :api_attr: imperative
    Decays the learning rate of ``optimizer`` by ``decay_rate`` once ``epoch`` reaches one of the milestones.
    The algorithm can be described as the code below.
    .. code-block:: text
        learning_rate = 0.5
        milestones = [30, 50]
        decay_rate = 0.1
        if epoch < 30:
            learning_rate = 0.5
        elif epoch < 50:
            learning_rate = 0.05
        else:
            learning_rate = 0.005
    Parameters:
        learning_rate (float|int): The initial learning rate. It can be set to python float or int number.
        milestones (tuple|list): List or tuple of each boundaries. Must be increasing.
        decay_rate (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * decay_rate`` .
            It should be less than 1.0. Default: 0.1.
    Returns:
        None.
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            with fluid.dygraph.guard():
                x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
                linear = fluid.dygraph.Linear(10, 10)
                input = fluid.dygraph.to_variable(x)
                scheduler = fluid.dygraph.MultiStepDecay(0.5, milestones=[3, 5])
                adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())
                for epoch in range(6):
                    for batch_id in range(5):
                        out = linear(input)
                        loss = fluid.layers.reduce_mean(out)
                        adam.minimize(loss)
                    scheduler.epoch()
                    print("epoch:{}, current lr is {}" .format(epoch, adam.current_step_lr()))
                    # epoch:0, current lr is 0.5
                    # epoch:1, current lr is 0.5
                    # epoch:2, current lr is 0.5
                    # epoch:3, current lr is 0.05
                    # epoch:4, current lr is 0.05
                    # epoch:5, current lr is 0.005
    """

    def __init__(self, learning_rate, milestones, decay_rate=0.1):
        if not isinstance(milestones, (tuple, list)):
            raise TypeError(
                "The type of 'milestones' in 'MultiStepDecay' must be 'tuple, list', but received %s."
                % type(milestones))

        if not all([
            milestones[i] < milestones[i + 1]
            for i in range(len(milestones) - 1)
        ]):
            raise ValueError('The elements of milestones must be incremented')
        if decay_rate >= 1.0:
            raise ValueError('decay_rate should be < 1.0.')

        self.milestones = milestones
        self.decay_rate = decay_rate
        super(MultiStepDecay, self).__init__(learning_rate)

    def get_lr(self):
        decay_rate = self.create_lr_var(self.decay_rate)
        for i in range(len(self.milestones)):
            if self.step_num < self.milestones[i]:
                return self.base_lr * (decay_rate ** i)

        return self.base_lr * (decay_rate ** len(self.milestones))