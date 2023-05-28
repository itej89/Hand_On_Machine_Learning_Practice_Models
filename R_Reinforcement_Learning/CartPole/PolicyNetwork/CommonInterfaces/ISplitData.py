import abc

class ISplitData(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'SplitData') and 
                callable(subclass.SplitData) and 
                NotImplemented)
    
    @abc.abstractclassmethod
    def split_data(self, _panda_data_frame):
        pass

