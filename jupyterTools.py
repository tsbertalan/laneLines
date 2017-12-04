import inspect
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter, LatexFormatter
from IPython.core.display import HTML

class src(object):
    
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

        