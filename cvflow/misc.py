import cvflow
import networkx, graphviz, matplotlib.pyplot as plt
import numpy as np

def _cached(method):
    def wrappedMethod(self, *args, **kwargs):
        if not hasattr(self, '__cache__'):
            self.__cache__ = {}
        key = '%s(*%s, **%s)' % (method.__name__, args, kwargs)
        if key in self.__cache__:
            out = self.__cache__[key]
        else:
            out = method(self, *args, **kwargs)
            self.__cache__[key] = out
        return out
    return wrappedMethod


def cached(method):
    return property(_cached(method))


class NodeDigraph:

    def __init__(self, format='png'):
        self._gv = graphviz.Digraph(format=format)
        self._nx = networkx.DiGraph()
        self.subgraphs = {}
        self.nodes = {}

    def __contains__(self, obj):
        return self._nid(obj) in self._nx

    def containsEdge(self, obj1, obj2):
        return (self._nid(obj1), self._nid(obj2)) in self._nx.edges

    def add_subgraph(self, members, baseName, graph_attr={}):
        baseName = 'cluster %s' % baseName
        siblings = self.subgraphs.get(baseName, [])
        l = len(siblings)
        label = graph_attr.get('label', '')
        if l > 0:
            name = '%s%d' % (baseName, l)
            label += ' %d' % (l+1,)
            graph_attr['label'] = label
        else:
            name = baseName
        sg = graphviz.Digraph(name=name, graph_attr=graph_attr)
        for member in members:
            self.add_node(member, gv=sg)
        siblings.append((name, sg))
        self._gv.subgraph(sg)
        self.subgraphs[baseName] = siblings
        return label

    def add_node(self, obj, gv=None):
        if gv is None:
            gv = self._gv
        if hasattr(obj, 'hidden') and obj.hidden:
            raise ValueError("Node %s is not supposed to be added to the graph." % obj)
        kw = {}
        if hasattr(obj, 'node_properties'):
            kw.update(obj.node_properties)
        label = str(obj)
        if obj.nodeName is not None:
            label = obj.nodeName
        kw['label'] = label
        nid = self._nid(obj)
        self._nx.add_node(nid)
        gv.node(nid, **kw)
        self.nodes[nid] = obj

    def _nid(self, obj):
        return ''.join((str(id(obj)) + str(obj)).split())

    def add_edge(self, obj1, obj2):
        for o in (obj1, obj2):
            self.add_node(o)
        n1 = self._nid(obj1)
        n2 = self._nid(obj2)
        if not self.containsEdge(obj1, obj2):
            self._gv.edge(n1, n2)
            self._nx.add_edge(n1, n2)

    def toposort(self):
        return [self.nodes[nid] for nid in networkx.topological_sort(self._nx)]


class Circle:

    def _defaultNodeProperties(self):
        return dict(shape='circle')


class Box:

    def _defaultNodeProperties(self):
        return dict(shape='box')


class Ellipse:

    def _defaultNodeProperties(self):
        return dict(shape='ellipse')


def isInteractive():
    """Are we in a notebook?"""
    import __main__ as main
    return not hasattr(main, '__file__')


def show(img, ax=None, title=None, clearTicks=True, titleColor='black', **subplots_adjust):
    """Display an image without x/y ticks."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img)
    if clearTicks:
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None: ax.set_title(title, color=titleColor)
    if len(subplots_adjust) > 0:
        ax.figure.subplots_adjust(**subplots_adjust)
    return ax.figure, ax


def axesGrid(count, fromsquare=True, preferTall=True, clearTicks=False, **subplotKwargs):

    # Either find the next-largest perfect square,
    # Or the divisors of count that are closest to its square root.
    a = int(np.ceil(np.sqrt(count)))
    if fromsquare:
        b = a
    else:
        for b in range(a, count+1):
            a = float(count) / b
            if a == int(a):
                a = int(a)
                break
    if (a-1) * b > count:
        a -= 1
    if preferTall:
        assert b >= a
        axes = plt.subplots(nrows=b, ncols=a, **subplotKwargs)[1]
    else:
        axes = plt.subplots(nrows=a, ncols=b, **subplotKwargs)[1]

    if clearTicks:
        for ax in axes.ravel():
            ax.patch.set_facecolor('None')
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])

    return axes
