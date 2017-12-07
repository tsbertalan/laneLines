import inspect, os
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter, LatexFormatter
from IPython.core.display import HTML, Latex, TextDisplayObject
from PIL import Image


class src(object):
    """Display the source code for an object in a pretty way in Jupyter."""
    
    def __init__(self, obj, start=0, end=-1, fname=True, **formatKwargs):
        self.obj = obj
        self.start = start
        self.end = end
        self.location = inspect.findsource(obj)[1]
        self.source = inspect.getsource(obj)
        self.fname = ''
        if fname:
            a = self.location + start + 1
            b = a + max(end, len(self.source.split('\n')) + end) - 1
            self.fname += 'Lines %d throuh %d of ' % (a, b)
            self.fname += os.path.basename(inspect.getsourcefile(obj))
            self.fname += ':'
            self.fname = '(%s)' % self.fname
        formatKwargs.setdefault('linenostart', self.location+start+1)
        formatKwargs.setdefault('linenos', 'inline')
        formatKwargs.setdefault('stripnl', False)
        self.formatKwargs = formatKwargs

    def _repr_latex_(self):
        return self.repr(LatexFormatter)
        
    def _repr_html_(self):
        return self.repr(HtmlFormatter, noclasses=True, **self.formatKwargs)

    def repr(self, Formatter, **kwargs):
        return self.fname + highlight(
            '\n'.join(
                self.source
                .split('\n')
                [self.start:self.end]
                ),
            PythonLexer(stripnl=False),
            Formatter(**kwargs)
        )


class GIFforLatex(TextDisplayObject):

    def __init__(self, gifpath):
        self.path = gifpath

    def _repr_html_(self):
        self.data = '<img src="%s" />' % self.path
        return self.data

    def __html__(self):
        """
        This method exists to inform other HTML-using modules (e.g. Markupsafe,
        htmltag, etc) that this object is HTML and does not need things like
        special characters (<>&) escaped.
        """
        return self._repr_html_()

    def _repr_latex_(self):
        im = Image.open(self.path)
        modpath = self.path.replace('.', '-').replace('_', '-') + '.png'
        im.save(modpath)
        self.data = r"\includegraphics{%s}" % modpath
        return self.data


def propertySrc(op, propertyName='value', **kw):
    """Unpack an @property and return a pretty-printing source-code object.
    
    from stackoverflow.com/questions/1360721
    """
    fg = op.__class__.mro()[0].__dict__[propertyName].fget
    import types
    if isinstance(fg, types.FunctionType):
        return src(fg, **kw)
    else:
        return src(fg.__closure__[0].cell_contents, **kw)

