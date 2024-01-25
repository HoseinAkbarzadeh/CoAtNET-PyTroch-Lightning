

class DotDict(dict):
    """ Dot notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = DotDict(**value)
            self[key] = value
    
    @classmethod
    def from_toml(cls, fpath):
        import toml
        with open(fpath, 'r') as f:
            config = toml.load(f)
        return cls(**config)
        
    def __getattribute__(self, __name):
        return super().__getattribute__(__name)
    
    def __setattr__(self, __name, __value):
        return super().__setattr__(__name, __value)