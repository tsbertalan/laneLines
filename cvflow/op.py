from functools import wraps
import numpy as np
import cvflow
from . import misc


class Prop:
    
    def __init__(self, implied=[], disimplied=[], dependents=[], **defaultNodeProperties):
        self.implied = implied
        self.disimplied = disimplied
        self.defaultNodeProperties = defaultNodeProperties
        self.dependents = dependents
        
    def __call__(self, method):
        propName = '_%s' % method.__name__
        @property
        def get(innerSelf):
            return getattr(innerSelf, propName, False)
        @get.setter
        def get(innerSelf, setValue):
            
            # Set the value itself.
            setattr(innerSelf, propName, setValue)
            
            # Set or unset the associated properties.
            if setValue:
                innerSelf._traits[propName] = dict(**self.defaultNodeProperties)
            else:
                innerSelf._traits.pop(propName, None)
                
            # Flip implied/disimplied flags.
            for imp in self.implied:
                if not getattr(innerSelf, imp) and setValue:
                    setattr(innerSelf, imp, setValue)
            for dependent in self.dependents:
                if getattr(innerSelf, dependent) and not setValue:
                    setattr(innerSelf, dependent, False)
            for dimp in self.disimplied:
                if getattr(innerSelf, dimp) and setValue:
                    setattr(innerSelf, dimp, False)

            setattr(innerSelf, propName, setValue)
            
        return get


def constantOrOp(obj):
    if not isinstance(obj, Op):
        from cvflow.baseOps import Constant
        obj = Constant(obj)
    return obj


def constantOrOpInput(method):
    def wrapped(self, input, *args, **kwargs):
        return method(self, constantOrOp(input), *args, **kwargs)
    return wrapped


class Op:

    def __init__(self):
        """

        In subclasses where this constructor is called as `super().__init__()`, 
        that call should come *after* the subclass is done initializting
        (especially setting its parents)
        """
        # Starting from the top of the MRO, accumulate node properties.
        for key in ('_visited', '_includeSelfInMembers', 'hidden'):
            setattr(self, key, getattr(self, key, False))

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
        """Should be called before any properties are set, like
        >>> self.isMono = True
        Since some are set by conveniece classes like `Mono`, 
        >>> super().__init__
        should always be called before these properties are set,
        so those properties can override super. Basically, we want
        addParent < super() < isMono
        to be the overriding order.
        """
        parent = constantOrOp(parent)
        if parent not in self.parents:
            self.parents.append(parent)
        if self not in parent.children:
            parent.children.append(self)

        self.copySetProperties(parent)

        return parent

    @misc.cached
    def value(self):
        raise NotImplementedError('`self.value` not implemented for class %s.' % type(self).__name__)

    def parent(self, index=0):
        try:
            return self.parents[index]
        except IndexError:
            raise IndexError('%s object has <= %d parents.' % (type(self).__name__, index))

    def child(self, index=0):
        try:
            return self.children[index]
        except IndexError:
            raise IndexError('%s object has <= %d children.' % (type(self).__name__, index))

    def walk(self, currentRecursionDepth=0, which=['parents', 'children'], excludeSelf=False):
        self._visited = True
        if not excludeSelf:
            yield self
        for k in which:
            l = dict(parents=self.parents, children=self.children)[k]
            for op in l:
                if not op._visited:
                    for op in op.walk(currentRecursionDepth=currentRecursionDepth+1):
                        yield op
        if currentRecursionDepth == 0:
            self._clearVisited()

    def getAncestors(self):
        for op in self.walk(self, which=['parents'], excludeSelf=True):
            yield op

    def getDescendants(self):
        for op in self.walk(which=['children'], excludeSelf=True):
            yield op

    def getRoot(self):
        roots = [root for root in self.getRoots(includeStatic=False)]
        assert len(roots) == 1, (roots, self)
        return list(roots)[0]

    # @generatorOrList
    def getRoots(self, includeStatic=False):
        for op in self.walk():
            if (includeStatic or not isinstance(op, cvflow.Static)) and len(op.parents) == 0:
                yield op

    # @generatorOrList
    def getByKind(self, Kind):
        for op in self.walk:
            if isinstance(op, Kind):
                yield op

    def assembleGraph(self, d=None, currentRecursionDepth=0, format='png', addKey=True, linkMultisteps=False):
        self._visited = True

        if d is None:
            d = misc.NodeDigraph(format=format)

        if self.hidden:
            assert len(self.parents) <= 1
        elif linkMultisteps or type(self) is not cvflow.Input:
            for parent in self.parents:
                parent = parent
                while parent.hidden and (linkMultisteps or not isinstance(parent, cvflow.Output)):
                    npar = len(parent.parents)
                    if npar == 0:
                        parent = None
                        break
                    else:
                        assert npar == 1, '%s has hidden=%s, but does not have 1 parent.' % (target, target.hidden)
                        parent = parent.parent()
                if (
                    parent is not None
                    and (linkMultisteps or type(parent) is not cvflow.Output)
                    ):
                    d.add_edge(parent, self)

        for parent in self.parents:
            if not parent._visited:
                parent.assembleGraph(d, currentRecursionDepth+1, linkMultisteps=linkMultisteps)

        for child in self.children:
            if not child._visited:
                child.assembleGraph(d, currentRecursionDepth+1, linkMultisteps=linkMultisteps)

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
                class L(cvflow.Logical): pass
                keyMembers = [
                    nn(Op, 'Continuous'),
                    nn(cvflow.Boolean, 'Binary'),
                    nn(cvflow.Mono, 'Single channel'),
                    nn(cvflow.Color, 'Three channel'),
                    nn(cvflow.PassThrough, 'No-op', dummy),
                    nn(cvflow.MultistepOp, 'multi-step result'),
                    nn(L, 'Logical')
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

    def draw(self, savePath=None, format='png', outType='graphviz', addKey=True, linkMultisteps=False):
        splitSubgraphs = not linkMultisteps
        d = self.assembleGraph(format=format, addKey=addKey, linkMultisteps=linkMultisteps)
        if savePath is not None:
            if savePath.lower().endswith('.%s' % format.lower()):
                savePath = savePath[:-4]
            if splitSubgraphs:
                graphObjects, nxobjs = d.subgraphs
                fpaths = [savePath + str(k) for k in range(len(graphObjects))]
            else:
                graphObjects = [d._gv]
                fpaths = [savePath]
            for graph, path in zip(graphObjects, fpaths):
                outPath = graph.render(path)
                print('Saved to %s.' % outPath)
        return dict(graphviz=d._gv, networkx=d._nx, NodeDigraph=d)[outType]

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    @property
    def nodeName(self):
        if hasattr(self, '_nodeName'):
            if isinstance(self._nodeName, str):
                return self._nodeName
            else:
                try:
                    return self._nodeName()
                except TypeError:
                    return str(self._nodeName)
        else:
            return None

    @nodeName.setter
    def nodeName(self, _nodeName):
        self._nodeName = _nodeName

    def getSimpleName(self, maxlen=17):
        if self.nodeName is not None:
            return self.nodeName

        if isinstance(self, cvflow.AsType) and len(self.parents) == 1:
            out = str(self.parent())
        elif isinstance(self, cvflow.PassThrough):
            out = '%s (from %s)' % (type(self).__name__, self.parent().getSimpleName())
        else:
            out = str(self)
        
        if out == '' or len(out) > maxlen:
            out = type(self).__name__

        # If it's still to long, just truncate.
        if len(out) > maxlen:
            out = out[:maxlen-3] + '...'

        return out

    def showValue(self, showMultistepParents=True, **kwargs):
        skipPastParentTypes = [cvflow.AsType]

        if isinstance(self, cvflow.MultistepOp):
            parents = [self.input]
            arrow = r'$\rightarrow\ldots\rightarrow$'
        else:
            arrow = r'$\rightarrow$'
            parents = [
                p for p in self.parents 
                if not isinstance(p, cvflow.Static)
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

    @constantOrOpInput
    def __and__(self, other):
        from cvflow.baseOps import And
        return And(self, other)

    @constantOrOpInput
    def __or__(self, other):
        from cvflow.baseOps import Or
        return Or(self, other)

    def __neg__(self):
        return cvflow.baseOps.ScalarMultiply(self, -1)

    def __invert__(self):
        return cvflow.baseOps.Not(self)

    @constantOrOpInput
    def __lt__(self, other):
        return cvflow.baseOps.LessThan(self, other)

    @constantOrOpInput
    def __le__(self, other):
        return cvflow.baseOps.LessThan(self, other, orEqualTo=True)

    @constantOrOpInput
    def __gt__(self, other):
        return cvflow.baseOps.GreaterThan(self, other)

    @constantOrOpInput
    def __ge__(self, other):
        return cvflow.baseOps.GreaterThan(self, other, orEqualTo=True)

    @property
    def _traits(self):
        k = '_Op__traits'
        setattr(self, k, getattr(self, k, {}))
        return getattr(self, k)
    
    @property
    def node_properties(self):
        out = {}
        for d in self._traits.values():
            out.update(d)
        return out

    def copySetProperties(self, other):
        propNames = ('isMono', 'isColor', 'isBoolean', 'isLogical', 'isPassThrough', 'isMultistepOp')
        notCopied = ('isLogical', 'isPassThrough', 'isMultistepOp')
        for propName in propNames:
            if propName not in notCopied:
                val = getattr(other, propName)
                if val:
                    setattr(self, propName, True)

    @Prop(disimplied=['isColor'], dependents=['isBoolean'], shape='box')
    def isMono(self): pass
    
    @Prop(disimplied=['isMono', 'isBoolean'], shape='box3d')
    def isColor(self): pass
    
    @Prop(implied=['isMono'], fontname='italic', color='gray', )
    def isBoolean(self): pass

    @Prop(style='dashed')
    def isLogical(self): pass

    @Prop(shape='none')
    def isPassThrough(self): pass

    @Prop(fontname='bold')
    def isMultistepOp(self): pass

    def assertProp(self, checkee, **kwargs):
        """
        Use like 
        >>> checker.assertProp(checkee, isMono=True)
        """
        assert len(kwargs) == 1, 'One kwarg should be passed to check!'
        propName = list(kwargs.keys())[0]
        propTarget = kwargs[propName]
        test = getattr(checkee, propName) == propTarget
        if not test:
            raise AssertionError('%s.%s on "%s" is not %s in %s.' % (
                type(checkee).__name__, propName, checkee.getSimpleName(), 
                propTarget, self.getSimpleName()
            ))


