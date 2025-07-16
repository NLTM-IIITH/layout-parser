from .global_utils import args
import numpy as np


def euclidean_distance(point1, point2):
    if args.metric == 'euclidean':
        squared_diff = (point1 - point2) ** 2
        sum_squared_diff = np.sum(squared_diff)
        distance = np.sqrt(sum_squared_diff)
    elif args.metric == 'chebyshev':
        x1, y1 = point1
        x2, y2 = point2

        # Calculate Chebyshev distance
        distance = np.max([np.abs(x1 - x2), np.abs(y1 - y2)])

    return distance

def euclidean_distance1(coord1, coord2):
    point1 = np.array(coord1)
    point2 = np.array(coord2)
    if args.metric == 'euclidean':
        squared_diff = (point1 - point2) ** 2
        sum_squared_diff = np.sum(squared_diff)
        distance = np.sqrt(sum_squared_diff)
    elif args.metric == 'chebyshev':
        x1, y1 = point1
        x2, y2 = point2

        # Calculate Chebyshev distance
        distance = np.max([np.abs(x1 - x2), np.abs(y1 - y2)])

    return distance


def minimum_euclidean(component):
    euclidean = float('inf')
    min_idx = -1
    for i in range(len(component)):
        if component['Visited'][i] != 1:
            current_distance = euclidean_distance(np.array([0, 0]), np.array(component['Top'][i]))
            if current_distance < euclidean:
                euclidean = current_distance
                min_idx = i
    return min_idx