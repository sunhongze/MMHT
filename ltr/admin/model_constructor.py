from functools import wraps
import importlib


def model_constructor(f):
    @wraps(f)
    def f_wrapper(*args, **kwds):
        net_constr = NetConstructor(f.__name__, f.__module__, args, kwds)
        output = f(*args, **kwds)
        if isinstance(output, (tuple, list)):
            # Assume first argument is the network
            output[0].constructor = net_constr
        else:
            output.constructor = net_constr
        return output
    return f_wrapper


class NetConstructor:
    def __init__(self, fun_name, fun_module, args, kwds):
        self.fun_name = fun_name
        self.fun_module = fun_module
        self.args = args
        self.kwds = kwds

    def get(self):
        net_module = importlib.import_module(self.fun_module)
        net_fun = getattr(net_module, self.fun_name)
        return net_fun(*self.args, **self.kwds)
