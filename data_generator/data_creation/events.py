import numpy

class Events:
    def __init__(self):
        self.x = None
        self.y = None
        self.t = None
        self.pol = None

    def add(self, x:numpy.ndarray, y:numpy.ndarray, t:numpy.ndarray, pol:numpy.ndarray):
        if self.x is None:
            self.x = x
        else:
            self.x = numpy.concatenate((self.x, x))
        if self.y is None:
            self.y = y
        else:
            self.y = numpy.concatenate((self.y, y))
        if self.t is None:
            self.t = t
        else:
            self.t = numpy.concatenate((self.t, t))
        if self.pol is None:
            self.pol = pol
        else:
            self.pol = numpy.concatenate((self.pol, pol))

    def reset(self):
        self.x = None
        self.y = None
        self.t = None
        self.pol = None

