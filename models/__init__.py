from .cycle_gan_model import  CycleGANModel
from .cycle_reid_model import CycleReidModel
def create_model(config):
    """
    :param config: The config
    :return: a mode according to the
    """
    if config.model == 'cycle_gan':
        model = CycleGANModel(config)
        return model
    elif config.model == 'cycle_gan_reid':
        model = CycleReidModel(config)
        return model
    elif config.model == 'test':
        raise NotImplementedError("The model has not been implemented")

    else:
        raise NotImplementedError("The model has not been implemented")