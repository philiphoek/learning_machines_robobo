import numpy

class Controller(object):


    def control(self, inputs, controller = None):

        action1 = numpy.random.choice([1,0])
        action2 = numpy.random.choice([1,0])

        return [action1, action2]