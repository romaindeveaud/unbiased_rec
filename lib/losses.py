import config


class Function(object):
  def __init__(self, raw_function):
    self.raw_function = raw_function

  def __call__(self, *args, **kwargs):
    return self.raw_function(*args, **kwargs)

  def __str__(self):
    return self.raw_function.__name__


@Function
def IPSMSELoss(output, target):
  return (((output - target)**2)/config.observation_propensities).mean()
