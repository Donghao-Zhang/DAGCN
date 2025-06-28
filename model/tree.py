"""
Basic operations on trees.
"""

import numpy as np
import copy
import sys
sys.path.append('../')


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def head_to_tree(head, len_, prune=-1, subj_pos=None, obj_pos=None, i=-1):
    """
    Convert a sequence of head indexes into a tree object.
    """
    head = head[:len_].tolist()
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1 # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h-1].add_child(nodes[i])
    else:
        # find dependency path
        subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
        obj_pos = [i for i in range(len_) if obj_pos[i] == 0]

        cas = None

        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h-1]
                subj_ancestors.add(h-1)
                h = head[h-1]

            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0:
                tmp += [h-1]
                obj_ancestors.add(h-1)
                h = head[h-1]
            cas.intersection_update(tmp)

        # find lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k:0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        try:
            path_nodes.add(lca)
        except:
            a=1

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

        for i in range(len_):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4) # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h-1] is not None
                nodes[h-1].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=False, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret


def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret


# 深度优先查找 返回从根节点到目标节点的路径
def deep_first_search(cur, val, path=[]):
    path.append(cur.idx)  # 当前节点值添加路径列表
    if cur.idx == val:  # 如果找到目标 返回路径列表
        return path

    if len(cur.children) == 0:
        return 'no'

    for node in cur.children:  # 对孩子列表里的每个孩子 进行递归
        t_path = copy.deepcopy(path)  # 深拷贝当前路径列表
        res = deep_first_search(node, val, t_path)
        if res == 'no':  # 如果返回no，说明找到头 没找到  利用临时路径继续找下一个孩子节点
            continue
        else:
            return res  # 如果返回的不是no 说明 找到了路径

    return 'no'  # 如果所有孩子都没找到 则 回溯


# 获取最短路径 传入两个节点值，返回结果，长度为节点的集合，包含了原始节点
def get_shortest_path(root, start, end):
    # 分别获取 从根节点 到start 和end 的路径列表，如果没有目标节点 就返回no
    path1 = deep_first_search(root, start, [])
    path2 = deep_first_search(root, end, [])
    if path1 == 'no' or path2 == 'no':
        return '无穷大', '无节点'
    # 对两个路径 从尾巴开始向头 找到最近的公共根节点，合并根节点
    len1, len2 = len(path1), len(path2)
    for i in range(len1 - 1, -1, -1):
        if path1[i] in path2:
            index = path2.index(path1[i])
            path2 = path2[index:]
            path1 = path1[-1:i:-1]
            break
    res = path1 + path2
    length = len(res)
    # path = '->'.join(res)
    # return '%s:%s' % (length, path)
    return length


def tree_to_distance(sent_len, tree):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    shortest path length is node number, which means that actural length should be (length - 1)
    """
    res = np.zeros((sent_len, sent_len), dtype=np.int)
    res_context = np.zeros((sent_len, sent_len), dtype=np.int)
    cur_max_len = sent_len-1
    for i in range(cur_max_len):
        if i >= cur_max_len:
            break
        for j in range(i+1, cur_max_len):
            if j >= cur_max_len:
                break
            length = get_shortest_path(tree, i, j)
            if isinstance(length, int):
                res[i, j] = res[j, i] = length
                res_context[i, j] = res_context[j, i] = j-i+1
            else:
                res[j:, :] = res[:, j:] = -1
                res_context[j:, :] = res_context[:, j:] = -1
                cur_max_len = j
                break
    for i in range(cur_max_len):
        res[i, i] = 1
        res_context[i, i] = 1
    return res_context, res
