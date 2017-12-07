import inspect
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter, LatexFormatter
from IPython.core.display import HTML, Latex, TextDisplayObject
from PIL import Image


class src(object):
    """Display the source code for an object in a pretty way in Jupyter."""
    
    def __init__(self, obj):
        self.obj = obj
        
    def _repr_latex_(self):
        return self.repr(LatexFormatter)
        
    def _repr_html_(self):
        return self.repr(HtmlFormatter, noclasses=True)

    def repr(self, Formatter, **kwargs):
        return highlight(
            inspect.getsource(self.obj),
            PythonLexer(),
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


def propertySrc(op, propertyName='value'):
    """Unpack an @property and return a pretty-printing source-code object.
    
    from stackoverflow.com/questions/1360721
    """
    return src(op.__class__.mro()[0].__dict__[propertyName].fget.__closure__[0].cell_contents)

