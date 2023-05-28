import abc

class ILoadData(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_Data') and 
                callable(subclass.load_Data) and
                NotImplemented)
    

    @abc.abstractclassmethod
    def load_Data(self):
        pass

