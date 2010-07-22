import numpy
import param, pattern
from topo.projection.basic import CFProjection, MaskedCFIter
from topo.base.patterngenerator import PatternGenerator

class CFProjectionM(CFProjection):
    """
    Modified CFProjection. It incorporates strength matrix - a mechanism that allows to specify strength
    of a projection for each neuron. The original strength is preserved, both strengths are multiplied.
    By default, this CFProjectionM behaves exactly like the normal projection (using constant pattern) as
    the default strength generator.
    """

    # Parameter that generates the strength matrix, could be any pattern..
    strength_generator = param.ClassSelector(PatternGenerator,
        default=pattern.Constant(),constant=False,
        doc="""Generator of projection's initial strength matrix.""")

    def __init__(self, **params):
        super(CFProjectionM,self).__init__(**params)
        # Initialize the pattern generator, so that the str. matrix has correct shape.
        self.strength_generator.xdensity, self.strength_generator.ydensity = self.activity.shape
        self.strength_matrix = self.strength_generator()

    def activate(self, input_activity):
        """
        Activate using the specified response_fn and output_fn.
        Multiply the activity with the strength matrix for each neuron in the same
        manner that projection strength multiplies the activity overall.
        """
        self.input_buffer = input_activity
        self.activity *=0.0
        self.response_fn(MaskedCFIter(self), input_activity, self.activity, self.strength)
        # Multiply the activity by the strength (element-wise, it is not a "matrix multiplication")
        self.activity *= self.strength_matrix

        for of in self.output_fns:
            of(self.activity)
        
