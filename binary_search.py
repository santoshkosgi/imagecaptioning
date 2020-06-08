"""
This file contains functions which analyse binary search and its variants.
"""
import math

def binary_search(array, target):
    l = 0
    h = len(array) - 1
    while l <= h:
        m = int((l+h)/2)
        if target > array[m]:
            l = m + 1
        elif target < array[m]:
            h = m - 1
        else:
            return m
    return "Not Found"


def less_than(array, target):
    l = 0
    h = len(array) - 1
    while l <= h:
        m = math.floor((l+h)/2)
        if target < array[m]:
            h = m - 1
        elif target > array[m]:
            l = m + 1
        else:
            h = m - 1
    return h


def greater_than(array, target):
    l = 0
    h = len(array) - 1
    while l <= h:
        m = math.floor((l+h)/2)
        if target < array[m]:
            h = m - 1
        elif target > array[m]:
            l = m + 1
        else:
            l = m + 1
    return l



print(less_than([1, 5, 7, 10, 13, 15], 0))
# print(binary_search([1,2,3,4,5], 6))