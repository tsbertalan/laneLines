import numpy as np
import cvflow
from . import misc


class Op:

    def __init__(self):
        if hasattr(self, '_defaultNodeProperties'):
            self.node_properties.update(self._defaultNodeProperties())
        self._visited = False
        self._includeSelfInMembers = False
        self.hidden = False

    @property
    def node_properties(self):
        if not hasattr(self, '_node_properties'):
            self._node_properties = {}
        return self._node_properties

    @node_properties.setter
    def node_properties(self, newdict):
        self._node_properties = newdict

    def invalidateCache(self):
        self.__cache__ = {}
        for child in self.children:
            child.invalidateCache()

    @property
    def parents(self):
        if not hasattr(self, '_parents'):
            self._parents = []
        return self._parents

    @property
    def children(self):
        if not hasattr(self, '_children'):
            self._children = []
        return self._children

    def addChild(self, child):
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parnts.append(self)
        return child

    def addParent(self, parent):
        if not isinstance(parent, Op):
            from cvflow.baseOps import Constant
            parent = Constant(parent)
        if parent not in self.parents:
            self.parents.append(parent)
        if self not in parent.children:
            parent.children.append(self)
        return parent

    @misc.cached
    def value(self):
        raise NotImplementedError

    def parent(self, index=0):
        return self.parents[index]

    def child(self, index=0):
        return self.children[index]

    def assembleGraph(self, d=None, currentRecursionDepth=0, format='png', addKey=True):
        # if isinstance(self, CvtColor):
        #     import utils; utils.bk()
        self._visited = True

        if d is None:
            d = misc.NodeDigraph(format=format)

        if self.hidden:
            assert len(self.parents) <= 1
        else:
            for parent in self.parents:
                target = parent
                while target.hidden:
                    npar = len(target.parents)
                    if npar == 0:
                        target = None
                        break
                    else:
                        assert npar == 1, '%s does not have 1 parent.' % target
                        target = target.parent()
                if target is not None:
                    d.add_edge(target, self)

        for parent in self.parents:
            if not parent._visited:
                parent.assembleGraph(d, currentRecursionDepth+1)

        for child in self.children:
            if not child._visited:
                child.assembleGraph(d, currentRecursionDepth+1)

        if currentRecursionDepth == 0:
            
            # Clear the visited flags so subsequent calls will work.
            self._clearVisited()

            # Add a key showing the styles of various node classes.
            if addKey:
                def nn(Cls, nodeName, *args):
                    op = Cls(*args)
                    op.nodeName = nodeName
                    return op

                dummy = cvflow.baseOps.Constant(42)
                dummy.hidden = True
                keyMembers = [
                    nn(Op, 'Continuous'),
                    nn(cvflow.Boolean, 'Binary'),
                    nn(cvflow.Mono, 'Single channel'),
                    nn(cvflow.Color, 'Tri-channel'),
                    nn(cvflow.PassThrough, 'No-op', dummy),
                    nn(cvflow.MultistepOp, 'multi-step result'),
                    #nn(cvflow.Constant, 'Constant', 42),
                ]

                d.add_subgraph(keyMembers, 'KEY', graph_attr=dict(color='gray', label='KEY'))

        return d

    def _clearVisited(self):
        self._visited = False
        for child in self.children:
            if child._visited:
                child._clearVisited()
        for parent in self.parents:
            if parent._visited:
                parent._clearVisited()

    def draw(self, savePath=None, format='png', outType='graphviz', addKey=True):
        d = self.assembleGraph(format=format, addKey=addKey)
        if savePath is not None:
            if savePath.lower().endswith('.%s' % format.lower()):
                savePath = savePath[:-4]
            outPath = d._gv.render(savePath)
            print('Saved to %s.' % outPath)
        return dict(graphviz=d._gv, networkx=d._nx, NodeDigraph=d)[outType]

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def getSimpleName(self, maxlen=17):
        if hasattr(self, 'nodeName'):
            return self.nodeName

        if isinstance(self, (
            #cvflow.PassThrough, 
            cvflow.AsType, cvflow.AsMono, cvflow.AsColor
            )) and len(self.parents) == 1:
            self = self.parent()
            out = str(self)
        elif isinstance(self, cvflow.PassThrough):
            out = '%s (%s)' % (type(self).__name__, self.parent().getSimpleName())
        else:
            out = str(self)
        
        if out == '' or len(out) > maxlen:
            out = type(self).__name__

        # If it's still to long, just truncate.
        if len(out) > maxlen:
            out = out[:maxlen-3] + '...'

        return out

    def showValue(self, showMultistepParents=True, **kwargs):
        excludedParentTypes = [cvflow.Constant, cvflow.CircleKernel]
        skipPastParentTypes = [cvflow.AsMono, cvflow.AsColor, cvflow.AsType]

        if isinstance(self, cvflow.MultistepOp):
            parents = [self.input]
            arrow = r'$\rightarrow\ldots\rightarrow$'
        else:
            arrow = r'$\rightarrow$'
            parents = [
                p for p in self.parents 
                if type(p) not in excludedParentTypes
            ]
            parents = [
                p.parent() if type(p) in skipPastParentTypes and len(p.parents) > 0 else p
                for p in parents
            ]

        parentNames = ', '.join([
            p.getSimpleName()
            for p in parents
        ])
        if len(parents) > 1: parentNames = '(%s)' % parentNames

        selfName = self.getSimpleName()
        kwargs.setdefault('title', r'%s %s %s' % (
            parentNames,
            arrow,
            selfName, 
        ))
        return misc.show(self.value, **kwargs)

    def checkType(self, obj, acceptedType, invert=False):
        # Wow it's almost like type-checking is a useful thing to have in a language.
        tname = acceptedType.__name__
        if acceptedType == cvflow.Mono:
            test = obj.isMono
        elif acceptedType == cvflow.Color:
            test = obj.isColor
        else:
            test = isinstance(obj, acceptedType)
        if invert:
            assert not test, '`%s` can\'t have %s nature for use in `%s`.' % (obj.getSimpleName(), tname, self)
        else:
            assert test, '`%s` needs to have %s nature for use in `%s`.' % (obj.getSimpleName(), tname, self)

    def getMembersByType(self, Kind, allowSingle=True):
        out = [m for m in self.members if isinstance(m, Kind)]
        if allowSingle and len(out) == 1:
            out = out[0]
        return out

    def __and__(self, other):
        from cvflow.baseOps import And
        return And(self, other)

    def __or__(self, other):
        from cvflow.baseOps import Or
        return Or(self, other)

    def __neg__(self):
        return cvflow.baseOps.ScalarMultiply(self, -1)

    def __invert__(self):
        return cvflow.baseOps.Not(self)

    @misc.cached
    def isMono(self):
        if isinstance(self, cvflow.Mono):
            return True
        elif isinstance(self, cvflow.Color):
            return False
        elif len(self.parents) > 0:
            return self.parent().isMono
        else:
            return False

    @misc.cached
    def isColor(self):
        if isinstance(self, cvflow.Color):
            return True
        elif isinstance(self, cvflow.Mono):
            return False
        elif len(self.parents) > 0:
            return self.parent().isColor
        else:
            return False
