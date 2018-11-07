
class DisplayItem(object):

    def __init__(self, name, default_value):
        self.name = name
        self.default_value = default_value

    def display_name_from_value(self, value) -> str:
        return ''

    def values_from_kwargs(self, **kwargs):
        values = kwargs.get(self.name, [self.default_value])
        assert isinstance(values, list)
        return values

    def value_from_kwargs(self, **kwargs):
        return kwargs.get(self.name, self.default_value)

    def update_kwargs_value(self, value, **kwargs):
        updated_kwargs = kwargs.copy()
        updated_kwargs.update({self.name: value})
        return updated_kwargs