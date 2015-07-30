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
            return Alpha(self.array[documentId])
        else:
            return self.array[:, documentId]



class UniformAlpha(Alpha):
    """A uniform alpha returns the same vector
    for every document"""

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
        
        
    