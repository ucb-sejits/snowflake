__author__ = 'nzhang-dev'

def partition(n, k):
    """
    :param n: Number to be partition
    :param k: number of partitions
    :return: generator of partition tuples
    """
    if k == 1:
        yield n,
        if n:
            yield -n,
        return
    for i in range(n+1):
        for part in partition(n-i, k-1):
            yield (i,) + part
            if i:
                yield (-i,) + part

def shell(n, k):
    """
    :param n: max norm requirement
    :param k: number of components
    :return: generator of tuples
    """
    return _shell(n, k, False)


def _shell(n, k, fulfilled=False):
    if k == 1:
        if not fulfilled:
            for i in (-n, n):
                yield i,
            return
        else:
            for i in range(-n, n+1):
                yield i,
            return
    for i in range(-n+1, n):
        for s in _shell(n, k-1, fulfilled):
            yield (i,) + s
    for i in (-n, n):
        for s in _shell(n, k-1, True):
            yield (i,) + s