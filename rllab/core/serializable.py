import inspect


class Serializable(object):

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        if getattr(self, "_serializable_initialized", False):
            return
        spec = inspect.getargspec(self.__init__)
        # Exclude the first "self" parameter
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        if spec.keywords:
            kwargs = locals_[spec.keywords]
        else:
            kwargs = dict()
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, "_serializable_initialized", True)

    def __getstate__(self):
        return {"__args": self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        # convert all __args to keyword-based arguments
        in_order_args = inspect.getargspec(self.__init__).args[1:]
        out = type(self)(**dict(zip(in_order_args, d["__args"]), **d["__kwargs"]))
        self.__dict__.update(out.__dict__)

    @classmethod
    def clone(cls, obj, **kwargs):
        assert isinstance(obj, Serializable)
        d = obj.__getstate__()
        d["__kwargs"] = dict(d["__kwargs"], **kwargs)
        out = type(obj).__new__(type(obj))
        out.__setstate__(d)
        return out
