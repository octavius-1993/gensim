#!/usr/bin/env python
#
# Allow support for alpha variants in LDA models

import numpy

class Alpha(object):
    """Alpha pretends to be a K by D array.
    Calling __getitem__ on an Alpha should produce
    a K-dimensional vector.
    In the basic class, a K by D array is required."""
    
    def __init__(self, nparray):
        """Provide a K by D array with the dth alpha in column d"""
        self.array = nparray

    def __getitem__(self, documentId):
        if isinstance(documentId, slice):
            return Alpha(self.array[:, documentId])
        else:
            return self.array[:, documentId]
        



class UniformAlpha(Alpha):
    """A uniform alpha returns the same vector
    for every document"""
    
    def __init__(self, nparray):
        """Provide a K length array"""
        self.array = nparray

    def __getitem__(self, documentId):
        if isinstance(documentId, slice):
            return self
        else:
            return self.array
    
class TwoPartAlpha(Alpha):
    """A two part alpha returns alpha_1 for documents 1, ..., D_1
    and alpha_2 for D_1 + 1, ..., D_1 + D_2"""
    
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
    """Provides a symmetric version of two part alpha"""
    
    def __init__(self, alpha1, alpha2, D1, K):
        """alpha1 and alpha2 are alpha values to be used
        before and after D1 respectively (first and second
        subcopora).
        K is the length of the alpha vectors (ie. num_topics)
        """
        
        self.array1 = numpy.asarray([alpha1]*K)
        self.array2 = numpy.asarray([alpha2]*K)
        self.cutoff = D1
        
        
if __name__ == '__main__':
    a = numpy.asarray([[1,2,3],[4,5,6]])
    alpha = Alpha(a)
    print(alpha[0])
    uniAlpha = UniformAlpha(a[0, :])
    print(uniAlpha[100])
    twoAlpha = TwoPartAlpha(a[0,:], a[1,:], 12)
    print(twoAlpha[0])
    print(twoAlpha[12])
    print(twoAlpha[11])
    
    print("Now test slice")
    print((alpha[0:2]).array)
    print(uniAlpha[0:100][40])
    print(twoAlpha[5:50][6])
    print(twoAlpha[5:50][7])
    
    print("Test symmetric two part")
    alpha2sym = TwoPartSymmetricAlpha(6, 7, 12, 10)
    print(alpha2sym[11])
    print(alpha2sym[12])
    