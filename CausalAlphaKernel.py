from elephant.kernels import AlphaKernel
import numpy as np

"""
Elephants AlphaKernel is not centered and hence not causal. 
https://github.com/NeuralEnsemble/elephant/issues/107

The feature branch by michael denker (https://github.com/INM-6/elephant/tree/feature/non-symmetric_instantaneous_rate)
was apparently never merged, but jpgill86 has posted this workaround in
the issue linked above 
"""

class CausalAlphaKernel(AlphaKernel):
    """
    This modified version of :class:`elephant.kernels.AlphaKernel` shifts time
    such that convolution of the kernel with spike trains (as in
    :func:`elephant.statistics.instantaneous_rate`) results in alpha functions
    that begin rising at the spike time, not before. The entire area of the
    kernel comes after the spike, rather than half before and half after, as
    with :class:`AlphaKernel <elephant.kernels.AlphaKernel>`. Consequently,
    CausalAlphaKernel can be used in causal filters.
    The equation used for CausalAlphaKernel is
    .. math::
        K(t) = \\left\\{\\begin{array}{ll} (1 / \\tau^2)
        \\ t\\ \\exp{(-t / \\tau)}, & t > 0 \\\\
        0, & t \\leq 0 \\end{array} \\right.
    with :math:`\\tau = \\sigma / \\sqrt{2}`, where :math:`\\sigma` is the
    parameter passed to the class initializer.
    In neuroscience a popular application of kernels is in performing smoothing
    operations via convolution. In this case, the kernel has the properties of
    a probability density, i.e., it is positive and normalized to one. Popular
    choices are the rectangular or Gaussian kernels.
    Exponential and alpha kernels may also be used to represent the postynaptic
    current / potentials in a linear (current-based) model.
    sigma : Quantity scalar
        Standard deviation of the kernel.
    invert: bool, optional
        If true, asymmetric kernels (e.g., exponential
        or alpha kernels) are inverted along the time axis.
        Default: False
    """

    def median_index(self, t):
        """
        In CausalAlphaKernel, "median_index" is a misnomer. Instead of
        returning the index into t that gives half area above and half below
        (median), it returns the index for the first non-negative time, which
        always corresponds to the start of the rise phase of the alpha
        function. This hack ensures that, when the kernel is convolved with a
        spike train, the entire alpha function is located to the right of each
        spike time.
        Overrides the following:
        """
        return np.nonzero(t >= 0)[0].min()
    # median_index.__doc__ += AlphaKernel.median_index.__doc__