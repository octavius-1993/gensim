#!/usr/bin/env python
#
# Allow support for alpha variants in LDA models

"""
Alpha objects facilitate the handling of alpha values in the mLDA model.

In mLDA, each document is allowed to have its own alpha value.
In most applications, only one or two distinct values are required.
The Alpha objects allow alpha values to be specified without bloating the data.
"""




import numpy

class Alpha(object):
    """Alpha acts like a K by D array.
    Calling ``__getitem__`` on an Alpha object should produce
    a K-dimensional vector.
    In the basic class, a K by D array is required."""
    
    def __init__(self, nparray):
        """Provide a K by D numpy array."""
        self.array = nparray

    def __getitem__(self, documentId):
        """Returns a K-dimensional vector, or a new Alpha object if sliced."""
        if isinstance(documentId, slice):
            return Alpha(self.array[:, documentId])
        else:
            return self.array[:, documentId]
        



class UniformAlpha(Alpha):
    """A uniform alpha returns the same alpha vector
    for every document"""
    
    def __init__(self, nparray):
        """Provide a K length array"""
        self.array = nparray

    def __getitem__(self, documentId):
        if isinstance(documentId, slice):
            return self
        else:
            return self.array
        
        
class UniformSymmetricAlpha(UniformAlpha):
    """A uniform symmetric alpha returns the same symmetric alpha for every document. The symmetric alpha is
    ``[alpha]*K``
    """
    def __init__(self, alpha, length):
        """Provide a (floating point) alpha value"""
        self.array = numpy.asarray([alpha]*length)
        self.alpha=alpha
    
    def __str__(self):
        return "Uniform symmetric alpha = %f" % self.alpha
    
class TwoPartAlpha(Alpha):
    """A two part alpha returns:
    - alpha_1 for documents 1, ..., D_1
    - alpha_2 for D_1 + 1, D_1 + 2, ...
    """
    
    def __init__(self, array1, array2, D1):
        """Provide the arrays for each subcorpus and the length of
        the first subcorpus"""
        self.array1 = array1
        self.array2 = array2
        self.cutoff = D1
        
    def __getitem__(self, documentId):
        if isinstance(documentId, slice):
            return TwoPartAlpha(self.array1, self.array2, self.cutoff - documentId.start)
        if documentId < self.cutoff:
            return self.array1
        else:
            return self.array2
        
class TwoPartSymmetricAlpha(TwoPartAlpha):
    """A two part symmetric alpha returns:
    - symmetric vector of alpha_1 for documents 1, ..., D_1
    - symmetric vector of alpha_2 for D_1 + 1, D_1 + 2, ...
    """
    
    def __init__(self, alpha1, alpha2, D1, K):
        """alpha1 and alpha2 are alpha values to be used
        before and after D1 respectively (first and second
        subcopora).
        K is the length of the alpha vectors (ie. num_topics)
        """
        
        self.array1 = numpy.asarray([alpha1]*K)
        self.array2 = numpy.asarray([alpha2]*K)
        self.cutoff = D1
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
    def __str__(self):
        return "Two part alpha split at %d, alpha1 = %f, alpha2 = %f" % (self.cutoff, self.alpha1, self.alpha2)    
        
        

    
