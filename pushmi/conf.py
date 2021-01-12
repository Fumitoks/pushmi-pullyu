class ConfigNamespace:
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return ConfigNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, ConfigNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else:
                setattr(self, key, val)

    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            d[key] = value.__dict__
        return d

def default_config():

    config_dict = {
    'dirs' : ConfigNamespace(),
    'arch' : ConfigNamespace(),
    'model' : ConfigNamespace(),
    'train' : ConfigNamespace(),
    }

    config = ConfigNamespace(**config_dict)

    config.dirs.root = '/content/gdrive/My Drive/PUSHMI/'
    config.dirs.data = '/content/gdrive/My Drive/PUSHMI/datasets/'
    config.dirs.saved_models = '/content/gdrive/My Drive/PUSHMI/saved_models/'
    config.dirs.logs = '/content/gdrive/My Drive/PUSHMI/logs/'
    config.dirs.py_drive = False

    config.arch.mode = 'SimpleConv'
    config.arch.channels = [1, 32, 64]
    config.arch.fc_dims = [7*7*64, 1000, 10]
    config.arch.kernel_size = [5]

    config.model.batch_size = 100   
    config.model.lr = 1e-5
    config.model.loss_type = 'distance'
    config.model.signature = ''

    config.train.max_epochs = 100
    config.train.val_check_interval = 0.2
    config.train.limit_train_batches = 1.
    config.train.limit_val_batches = 1.
    config.train.accumulate_grad_batches = 1

    return config