import abc

class IPlotData(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'PlotHistogram') and 
                callable(subclass.PlotHistogram) and 
                hasattr(subclass, 'PlotColumnHistogram') and 
                callable(subclass.PlotColumnHistogram) and
                NotImplemented)
    
    @abc.abstractclassmethod
    def PlotHistogram(self, _panda_data_frame):
        pass

    @abc.abstractclassmethod
    def PlotColumnHistogram(self, _panda_data_frame, ColumnID):
        pass
        
