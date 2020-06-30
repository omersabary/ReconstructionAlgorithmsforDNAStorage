import numpy as np
import operator
from edit_distance import edit_distance_ops
from all_lcs import all_lcs
import sys


def compare_distance(s, n):
    return abs(len(s) - n)


def sort_cluster(cluster, n):
    return sorted(cluster, key=lambda x: abs(len(x) - n))


# determine an index for each lcs character according to to copies in the cluster
def align_lcs(cluster, lcs_string, orig_len):
    error_vectors = [edit_distance_ops(c, lcs_string, format_list=True)[1] for c in
                     cluster]  # calculate the error vector between each copy to the lcs
    votes_vector = [[0] * (orig_len + 1) for x in
                    range(len(lcs_string))]  # votes histogram : size of lcs string * length of target string
    # fill votes table
    for ev in error_vectors:
        # print(ev)
        n = len(ev)
        m = len(lcs_string)
        i = 0  # points to beginning of error vector (copy)
        j = 0  # points to beginning of the lcs string
        k = 0  # points to current index in target word
        while i < n and j < m and k < orig_len:
            if ev[i][0] == 'I':
                i += 1
                j += 1  # if we insert a char to the copy char is already at lcs so catch up to it
                # k += 1
            elif ev[i][0] == 'D':
                i += 1
                k += 1
            elif ev[i][0] == 'R':
                votes_vector[j][-1] += 1
                k += 1
                i += 1
                j += 1
            elif ev[i][0] == 'H':
                temp = votes_vector[j]
                try:
                    temp[k] += 1
                except IndexError:
                    print("i: {}, j: {}, k: {}".format(i, j, k))
                    exit()
                i += 1
                j += 1
                k += 1
            else:
                raise Exception("WTF")
        if j < m:  # there are lcs characters that don't exists in the copy
            for k in range(j, m):
                votes_vector[k][-1] += 1
    # for l in votes_vector:
    #     print(l)
    indices = []
    for i in range(len(votes_vector)):
        l_np = np.asarray(votes_vector[i])
        index = l_np.argmax() if l_np.argmax() < orig_len else -1
        indices.append(index)
    return indices


def complete_missing(cluster, target_string, orig_len):
    error_vectors = [edit_distance_ops(c, target_string, format_list=True)[1] for c in
                     cluster]  # calculate the error vector between each copy to the lcs
    aligned_cluster = []
    for ev in error_vectors:
        n = len(ev)
        i = 0  # points to beginning of error vector (copy)
        s = ''
        while i < n:
            if ev[i][0] == 'I':
                s += ev[i][1] if ev[i][1] != '-' else '.'  # '.' means vote nothing
                i += 1
            elif ev[i][0] == 'D':
                i += 1
            elif ev[i][0] == 'R':
                s += ev[i][2] if ev[i][2] != '-' else ev[i][1]
                i += 1
            elif ev[i][0] == 'H':
                s += ev[i][1]
                i += 1
            else:
                raise Exception("WTF")
        assert orig_len == len(s)
        aligned_cluster.append(s)
        # print(s)
    for i in range(len(target_string)):
        if target_string[i] == '-':
            h = {'A': 0, 'C': 0, 'G': 0, 'T': 0, '.': -np.infty}
            for ac in aligned_cluster:
                try:
                    h[ac[i]] += 1
                except KeyError:
                    print("found the char {} in one of the copies!".format(ac[i]))
                    continue
            target_string[i] = max(h.items(), key=operator.itemgetter(1))[0]
    return target_string


def majority(copies, orig_len):
    for copy in copies:
        assert len(copy) == orig_len
    res_string = ['-'] * orig_len
    for i in range(orig_len):
        votes = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        for copy in copies:
            votes[copy[i]] += 1
        res_string[i] = max(votes.items(), key=operator.itemgetter(1))[0]
    return res_string


def Alg(cluster, orig_len: int):
    # sort cluster by length (closest to n first)
    # if len(cluster) < 2:
    # return "cluster size must be more than 2!"
    # return cluster[0]
    assert (len(cluster) >= 2)
    targets = []
    cluster = sort_cluster(cluster, orig_len)
    li = [num for num in range(len(cluster))]
    # list_of_pairs = [(li[p1], li[p2]) for p1 in range(len(li)) for p2 in range(p1 + 1, len(li))]
    # for pair in list_of_pairs:
    #     print("checking for pair: ", pair)
    for random_lcs in all_lcs(cluster[0], cluster[1]):
        # lcs_num = 0
        # for random_lcs in all_lcs(cluster[pair[0]], cluster[pair[1]]):
        #     if lcs_num > 20:
        #         break
        #     lcs_num += 1
        target_str = ['-'] * orig_len
        indices = align_lcs(cluster, random_lcs, orig_len)
        # print(indices)
        for i in range(len(random_lcs)):
            target_str[indices[i]] = random_lcs[i]
        target_str = complete_missing(cluster, target_str, orig_len)
        targets.append(target_str)
    return "".join(majority(targets, orig_len))


def run_alg(cluster, orig_len):
    sys.setrecursionlimit(2000)
    if len(cluster) < 2:
        return cluster[0]
    cluster_reversed = [c[::-1] for c in cluster]

    strait_ans = Alg(cluster, orig_len)
    if strait_ans[0] == 'c':
        return strait_ans
    reversed_ans = Alg(cluster_reversed, orig_len)[::-1]
    #
    ans = strait_ans[:orig_len // 2] + reversed_ans[orig_len // 2:]
    target_str = ['-'] * orig_len
    indices = align_lcs(cluster, ans, orig_len)
    for i in range(len(target_str)):
        target_str[indices[i]] = ans[i]

    # return strait_ans, reversed_ans, ans
    return ans
    # return strait_ans
