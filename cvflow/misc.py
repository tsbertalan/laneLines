import networkx, graphviz, matplotlib.pyplot as plt

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

    def __contains__(self, obj):
        return self._nid(obj) in self._nx

    def containsEdge(self, obj1, obj2):
        return (self._nid(obj1), self._nid(obj2)) in self._nx.edges

    def add_node(self, obj):
        # if isinstance(obj, Constant):
        #     import utils; utils.bk()
        if hasattr(obj, '_skipForPlot') and obj._skipForPlot:
            raise ValueError("Node %s is not supposed to be added to the graph." % obj)
        if obj in self:
            return
        kw = {}
        if hasattr(obj, 'node_properties'):
            kw.update(obj.node_properties)
        kw['label'] = str(obj)
        #kw.setdefault('label', str(obj))
        nid = self._nid(obj)
        self._gv.node(nid, **kw)
        self._nx.add_node(nid)

    def _nid(self, obj):
        return ''.join((str(id(obj)) + str(obj)).split())

    def add_edge(self, obj1, obj2):
        for o in (obj1, obj2):
            if o not in self:
                self.add_node(o)

        n1 = self._nid(obj1)
        n2 = self._nid(obj2)
        if not self.containsEdge(obj1, obj2):
            self._gv.edge(n1, n2)
            self._nx.add_edge(n1, n2)


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


def show(img, ax=None, title=None, clearTicks=True):
    """Display an image without x/y ticks."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img)
    if clearTicks:
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None: ax.set_title(title)
    return ax.figure, ax
