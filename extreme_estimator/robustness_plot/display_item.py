
class DisplayItem(object):

    def __init__(self, argument_name, default_value, dislay_name=None):
        self.argument_name = argument_name
        self.default_value = default_value
        self.dislay_name = dislay_name if dislay_name is not None else self.argument_name

    def values_from_kwargs(self, **kwargs):
        values = kwargs.get(self.argument_name, [self.default_value])
        assert isinstance(values, list)
        return values

    def value_from_kwargs(self, **kwargs):
        return kwargs.get(self.argument_name, self.default_value)

    def update_kwargs_value(self, value, **kwargs):
        updated_kwargs = kwargs.copy()
        updated_kwargs.update({self.argument_name: value})
        return updated_kwargs