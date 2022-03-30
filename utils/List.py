from diploma import *
class List(list):
    def __init__(self, *args, **kwargs):
        super(List, self).__init__(args[0])
    def __add__(self, x):
        return List(super().__add__(x))
    def filter(self,callback):
        self = List(filter(callback, self))
        return self
    def reduce(self,callback,init_val):
        return reduce(callback,self,init_val)
    def map(self,callback):
        self = List(map(callback, self))
        return self
    def find(self,callback):
        for val in self:
            if callback(val):
                 return val
    def findIndex(self,callback):
        for index,val in enumerate(self):
            if callback(val):
                 return index
    def len(self):
        return len(self)
    def unique(self,return_counts=False):
        return List(np.unique(self,return_counts=return_counts))
    def shuffle(self):
        return List(shuffle(self))
    def sort(self, key=None,reverse=False):
        self = List(sorted(self,key=key,reverse=reverse))
        return self
    def first(self):
        return self[0]
    def flatten(self):
        self = List(item for sublist in self for item in sublist)
        return self

