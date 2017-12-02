class Nature:
    pass

class Mono(Nature):
    pass

class Color(Nature):
    pass

class Binary(Mono):
    pass


class Natured:

    @property
    def _natures(self):
        if not hasattr(self, '__natures'): self.__natures = set()
        return self.__natures

    def hasNature(self, nature):
        return nature in self._natures

    def addNature(self, nature):

        default_properties = dict(
            Mono=dict(shape='box'),
            Color=dict(shape='box3d'),
            Binary=dict(style='dashed'),
        )

        if nature == 'mono':
            self.node_properties.update()

        self._natures.add(nature)