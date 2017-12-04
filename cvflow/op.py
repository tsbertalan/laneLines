from functools import wraps
import numpy as np
import cvflow
from . import misc


class Prop:
    
    def __init__(self, implied=[], disimplied=[], dependents=[], default=False, **defaultNodeProperties):
        self.implied = implied
        self.disimplied = disimplied
        self.defaultNodeProperties = defaultNodeProperties
        self.dependents = dependents
        self.default = default
        
    def __call__(self, method):
        propName = '_%s' % method.__name__
        @property
        def get(innerSelf):
            return getattr(innerSelf, propName, self.default)
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
    def wrapped(self, *args, **kwargs):
        return method(self, 
            *[
                constantOrOp(arg)
                for arg in args
            ],
            **kwargs
        )
    return wrapped


def scalarOutputIfScalarInputs(method):
    def wrapped(self, *args, **kwargs):
        out = method(self, *args, **kwargs)
        if all([isinstance(arg, (int, float)) or arg.isScalar for arg in args]) and self.isScalar:
            out.isScalar = True
        return out
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
        misc.clearCache(self)
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

    @misc.cached()
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

    def _walk(self, currentRecursionDepth=0, which=['parents', 'children'], excludeSelf=False):
        self._visited = True
        if not excludeSelf:
            yield self
        for k in which:
            l = dict(parents=self.parents, children=self.children)[k]
            for op in l:
                if not op._visited:
                    for op in op.walk(
                            currentRecursionDepth=currentRecursionDepth+1, 
                            which=which, excludeSelf=False
                        ):
                        yield op
        if currentRecursionDepth == 0:
            self._clearVisited()

    def walk(self, **kwargs):
        # To prevent callers from stopping early and leaving the tree
        # in some undermined state v/v _visited, we need to make a full list.
        # TODO: Implement walk as an object with a __del__ method that calls _clearVisited().
        return [op for op in self._walk(**kwargs)]

    def getAncestors(self):
        for op in self.walk(which=['parents'], excludeSelf=True):
            yield op

    def getDescendants(self):
        for op in self.walk(which=['children'], excludeSelf=True):
            yield op

    def getRoot(self):
        roots = [root for root in self.getRoots(includeStatic=False)]
        assert len(roots) == 1, (roots, self)
        return list(roots)[0]

    def getRoots(self, includeStatic=False):
        for op in self.walk():
            if (includeStatic or not isinstance(op, cvflow.Static)) and len(op.parents) == 0:
                yield op

    def getByKind(self, Kind, index=None, asList=True, **kwargs):
        if index is None:
            out = self._getByKindGenerator(Kind, **kwargs)
            if asList:
                out = list(out)
            return out
        else:
            i = 0
            for op in self._getByKindGenerator(Kind, **kwargs):
                if i == index:
                    return op
                i += 1

    def _getByKindGenerator(self, Kind, which='all'):
        source = {
            'all': self.walk, 
            'ancestors': self.getAncestors, 
            'descendants': self.getDescendants
        }[which]
        for op in source():
            if isinstance(op, Kind):
                yield op

    @property
    def numVisibleParents(self):
        return len([p for p in self.parents if not p.hidden])

    def assembleGraph(self, d=None, currentRecursionDepth=0, format='png', addKey=True, linkMultisteps=False):
        self._visited = True

        if d is None:
            d = misc.NodeDigraph(format=format)

        if self.hidden:
            assert self.numVisibleParents <= 1
        elif linkMultisteps or type(self) is not cvflow.Input:
            for parent in self.parents:
                parent = parent
                while parent.hidden and (linkMultisteps or not isinstance(parent, cvflow.Output)):
                    npar = parent.numVisibleParents
                    if npar == 0:
                        parent = None
                        break
                    else:
                        assert npar == 1, '%s has hidden=%s, but does not have 1 parent.' % (parent, parent.hidden)
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
            graphObjects = [d._gv]
            fpaths = [savePath]
            if splitSubgraphs:
                graphObjects.extend(d.subgraphs[0])
                fpaths.extend([savePath + str(k) for k in range(len(graphObjects))])
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
        title = r'%s %s %s' % (
            parentNames,
            arrow,
            selfName, 
        )
        if len(title) > 50:
            title = selfName
        kwargs.setdefault('title', title)
        return misc.show(self.value, **kwargs)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __truediv__(self, other):
        return cvflow.baseOps.Divide(self, other, zeroSpotsToMax=False)

    @constantOrOpInput
    def __floordiv__(self, other):
        return cvflow.baseOps.Divide(self, other, zeroSpotsToMax=True)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __mul__(self, other):
        return cvflow.baseOps.Multiply(self, other)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __abs__(self):
        return cvflow.baseOps.Abs(self)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __add__(self, other):
        return cvflow.baseOps.Add(self, other)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __sub__(self, other):
        return cvflow.baseOps.Subtract(self, other)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __pow__(self, power):
        return cvflow.Pow(self, power)

    def max(self):
        return cvflow.Max(self)
    
    def min(self):
        return cvflow.Min(self)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __and__(self, other):
        return cvflow.baseOps.And(self, other)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __or__(self, other):
        return cvflow.baseOps.Or(self, other)

    @scalarOutputIfScalarInputs
    def __neg__(self):
        return cvflow.baseOps.ScalarMultiply(self, -1)

    @scalarOutputIfScalarInputs
    def __invert__(self):
        return cvflow.baseOps.Not(self)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __lt__(self, other):
        return cvflow.baseOps.LessThan(self, other)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __le__(self, other):
        return cvflow.baseOps.LessThan(self, other, orEqualTo=True)

    @scalarOutputIfScalarInputs
    @constantOrOpInput
    def __gt__(self, other):
        return cvflow.baseOps.GreaterThan(self, other)

    @scalarOutputIfScalarInputs
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

    @Prop()# TODO: Get some properties for this.
    def isArithmetic(self): pass

    @Prop(shape='none')
    def isPassThrough(self): pass

    @Prop(fontname='bold')
    def isMultistepOp(self): pass

    @Prop()
    def isScalar(self): pass

    @Prop(default=True)
    def isVisualized(self): pass

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

    @misc.cached()
    def shape(self):
        if len(self.parents) > 0:
            shapes = [parent.shape for parent in self.parents if parent.shape is not None]
            if len(shapes) > 0:
                return shapes[0]
    @shape.setter
    def shape(self, newshape):
        self.__cache__ = getattr(self, '__cache__', {})
        self.__cache__[cvflow.misc.cacheKey('shape', newshape)] = newshape

    @constantOrOpInput
    def constructColorOutpout(self, *args, dtype='uint8', scaleUintTo255=True):
        # 'zeros' can be passed for some of the args, but only if
        # self.input.shape is defined.
        # TODO: handle this direcly in ColorJoin; maybe by passing a reference to self. Or by taking the shape of a nonzero parent.value, and degrading to a zero-column if they're all zeros.
        channels = [
            cvflow.Constant(np.zeros(self.input.shape).astype(dtype))
            if getattr(c, 'theConstant', None) == 'zeros'
            else cvflow.AsType(c, dtype, scaleUintTo255=scaleUintTo255)
            for c in args
        ]
        for c in channels:
            if isinstance(c, cvflow.Constant):
                c.hidden = True
        self.colorOutput = cvflow.ColorJoin(*channels)
        self.members = [self.colorOutput]
        return self.colorOutput

    def nparent(self, n):
        out = self
        for i in range(n):
            out = out.parent()
        return out
