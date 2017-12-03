import cvflow
import networkx, graphviz, matplotlib.pyplot as plt
import numpy as np

def cacheKey(objname, *args, **kwargs):
    return '%s(*%s, **%s)' % (objname, args, kwargs)


def _cached(method):
    def wrappedMethod(self, *args, **kwargs):
        if not hasattr(self, '__cache__'):
            self.__cache__ = {}
        key = cacheKey(method.__name__, *args, **kwargs)
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
        self._subgraphs = {}
        self.nodes = {}

    def __contains__(self, obj):
        return self._nid(obj) in self._nx

    def containsEdge(self, obj1, obj2):
        return (self._nid(obj1), self._nid(obj2)) in self._nx.edges

    @property
    def subgraphs(self):
        gvs = sum([[y[1] for y in x] for x in self._subgraphs.values()], [])
        nxs = sum([[y[2] for y in x] for x in self._subgraphs.values()], [])
        return gvs, nxs

    def add_subgraph(self, members, baseName, graph_attr={}):
        baseName = 'cluster %s' % baseName
        siblings = self._subgraphs.get(baseName, [])
        l = len(siblings)
        label = graph_attr.get('label', '')
        if l > 0:
            name = '%s%d' % (baseName, l)
            label += ' %d' % (l+1,)
            graph_attr['label'] = label
        else:
            name = baseName
        gv = graphviz.Digraph(name=name, format=self._gv.format, graph_attr=dict(graph_attr))
        graph_attr.pop('label', None)
        gvd = graphviz.Digraph(name=name, format=self._gv.format, graph_attr=dict(graph_attr))
        nx = networkx.DiGraph()
        for member in members:
            self.add_node(member, gvnx=(gv, nx))
            kw = self.getKw(member)
            n1 = self._nid(member)
            gvd.node(n1, **kw)
            self.add_node(member, gvnx=(gvd, nx))
            for child in members:
                if self.containsEdge(member, child):
                    n2 = self._nid(child)
                    gvd.edge(n1, n2)
        siblings.append((name, gvd, nx))
        self._gv.subgraph(gv)
        self._subgraphs[baseName] = siblings
        return label

    def getKw(self, obj):
        # Assemble the styling properties.
        kw = {}
        if hasattr(obj, 'node_properties'):
            kw.update(obj.node_properties)
        label = str(obj)
        if obj.nodeName is not None:
            label = obj.nodeName
        kw['label'] = label
        return kw

    def add_node(self, obj, gvnx=None):
        # Maybe fill subgraphs?
        if gvnx is None:
            gv = self._gv
            nx = self._nx
        else:
            gv, nx = gvnx

        if hasattr(obj, 'hidden') and obj.hidden:
            raise ValueError("Node %s is not supposed to be added to the graph." % obj)

        kw = self.getKw(obj)

        # Inser the node by its NID.
        nid = self._nid(obj)
        nx.add_node(nid)
        gv.node(nid, **kw)

        self.nodes[nid] = obj

    def _nid(self, obj):
        return ''.join((str(id(obj)) + str(obj)).split())

    def add_edge(self, obj1, obj2, gvnx=None):
        if gvnx is None:
            gv = self._gv
            nx = self._nx
        else:
            gv, nx = gvnx
        for o in (obj1, obj2):
            self.add_node(o, gvnx=gvnx)
        n1 = self._nid(obj1)
        n2 = self._nid(obj2)
        if gvnx is not None or not self.containsEdge(obj1, obj2):
            self._gv.edge(n1, n2)
            self._nx.add_edge(n1, n2)

    def toposort(self):
        return [self.nodes[nid] for nid in networkx.topological_sort(self._nx)]


def isInteractive():
    """Are we in a notebook?"""
    import __main__ as main
    return not hasattr(main, '__file__')


def show(img, 
    ax=None, title=None, clearTicks=True, 
    titleColor='black', histogramOverlayAlpha=0, 
    doLegend=True,
    **subplots_adjust):
    """Display an image without x/y ticks."""
    # Make axes for plotting if we weren't supplied some from outside.
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the actual image.
    ax.imshow(img)

    # Maybe draw one or three histograms.
    mono = len(img.shape) == 2 or img.shape[-1] == 1
    getData = lambda ichannel: np.copy(img.reshape((img.shape[0], img.shape[1], -1))[:, :, ichannel]).ravel()
    boolean = all([len(set(getData(k))) <= 2 for k in range(1 if mono else 3)])
    didvline = []
    if histogramOverlayAlpha:
        hi = img.shape[0]
        def hist(ichannel, color, heightFraction=.25, histAlpha=histogramOverlayAlpha):
            
            if not mono:
                histAlpha /= 2

            def normalize(vec):
                vec -= vec.min()
                if vec.max() != 0:
                    vec = vec / vec.max()
                return vec

            data = normalize(getData(ichannel))

            # Generate the rescaled histogram.
            hist, bins = np.histogram(data, bins=128 if not boolean else 2)
            # Use bin centers.
            bins = (bins[:-1] + bins[1:]) / 2.
            imax = np.argmax(hist)
            vmax = int(255*bins[imax])
            bins = normalize(bins)
            hist = normalize(np.log10(hist + 1e-10))

            hist *= img.shape[0] * heightFraction
            bins *= img.shape[1]

            # Plot the histogram.
            ax.fill_between(bins, hi, hi-hist, alpha=histAlpha, color=color, zorder=999)

            # Add a vline at the mode.
            ax.axvline(bins[imax], label='mode: %d' % vmax, color=color)
            didvline.append(1)

            # Add a vline at the largest value with some presense in the data.
            if mono:
                brightest = max(bins[hist/max(hist)>1e-4])
                ax.axvline(
                    brightest, 
                    label='~max: %d' % brightest, color=color, linestyle='--'
                )
        
        # Don't bother trying to make histograms for boolean data.
        if not boolean:
            if mono:
                hist(0, 'white')
            else:
                # For tricolor images, pretend the colors are RGB and histogram accordingly.
                for i in range(3):
                    hist(i, ['red', 'green', 'blue'][i])

    # Doctor up the plot a bit.
    if clearTicks:
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None: ax.set_title(title, color=titleColor)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    if doLegend and didvline: ax.legend(fontsize=6, loc='upper right')
    if len(subplots_adjust) > 0:
        ax.figure.subplots_adjust(**subplots_adjust)


    return ax.figure, ax


def clearTicks(ax):
    ax.patch.set_facecolor('None')
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])


def axesGrid(count, fromsquare=True, preferTall=True, clearAxisTicks=False, **subplotKwargs):

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

    if clearAxisTicks:
        for ax in axes.ravel():
            clearTicks(ax)
            

    return axes



class VisualizeFilter:
    def __init__(self, colorFilter, allFrames):
        self.colorFilter = colorFilter
        self.allFrames = allFrames
    
    
    def __call__(self, frame, clearAxes=True, closeFigure=False, **kwargs):
        import utils
        self.colorFilter(frame)
        axes = getattr(self, 'axes', None)
        figs = self.colorFilter.showMembers(
            axes=axes, subplotKwargs=dict(figsize=(16,9)), 
            wspace=0, showMultistepParents=False, **kwargs
        )
        if len(figs) > 0:
            fig = figs[0]
            self.axes = fig.axes
            for ax in fig.axes:
                cvflow.misc.clearTicks(ax)
            out = utils.fig2img(fig)
            if clearAxes:
                for ax in fig.axes:
                    ax.cla()
            if closeFigure:
                plt.close(fig)
            return out
    
    @property
    def fig(self):
        return self.axes[0].figure
    
    def visualizeFromKey(self, k, maxFrames=np.inf, **kwargs):
        import utils
        frames = self.allFrames[k]
        fpath = 'cf-%s-%s-vis' % (self.colorFilter, k)
        if maxFrames < np.inf:
            frames = frames[:maxFrames]
            fpath += '-%dframes' % maxFrames
        fpath += '.mp4'
        vid = utils.transformVideo(frames, fpath, lambda frame: self(frame, **kwargs), desc=fpath)
        self(frames[-1], clearAxes=False)
        return vid, self.fig
        
    def __del__(self):
        if hasattr(self, 'axes'):
            plt.close(self.fig)
