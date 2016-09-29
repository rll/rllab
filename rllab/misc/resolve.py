from pydoc import locate
import types
from rllab.misc.ext import iscanr


def classesinmodule(module):
    md = module.__dict__
    return [
        md[c] for c in md if (
            isinstance(md[c], type) and md[c].__module__ == module.__name__
        )
    ]


def locate_with_hint(class_path, prefix_hints=[]):
    module_or_class = locate(class_path)
    if module_or_class is None:
        # for hint in iscanr(lambda x, y: x + "." + y, prefix_hints):
        #     module_or_class = locate(hint + "." + class_path)
        #     if module_or_class:
        #         break
        hint = ".".join(prefix_hints)
        module_or_class = locate(hint + "." + class_path)
    return module_or_class
   

def load_class(class_path, superclass=None, prefix_hints=[]):
    module_or_class = locate_with_hint(class_path, prefix_hints)
    if module_or_class is None:
        raise ValueError("Cannot find module or class under path %s" % class_path)
    if type(module_or_class) == types.ModuleType:
        if superclass:
            classes = [x for x in classesinmodule(module_or_class) if issubclass(x, superclass)]
        if len(classes) == 0:
            if superclass:
                raise ValueError('Could not find any subclasses of %s defined in module %s' % (str(superclass), class_path))
            else:
                raise ValueError('Could not find any classes defined in module %s' % (class_path))
        elif len(classes) > 1:
            if superclass:
                raise ValueError('Multiple subclasses of %s are defined in the module %s' % (str(superclass), class_path))
            else:
                raise ValueError('Multiple classes are defined in the module %s' % (class_path))
        else:
            return classes[0]
    elif isinstance(module_or_class, type):
        if superclass is None or issubclass(module_or_class, superclass):
            return module_or_class
        else:
            raise ValueError('The class %s is not a subclass of %s' % (str(module_or_class), str(superclass)))
    else:
        raise ValueError('Unsupported object: %s' % str(module_or_class))
