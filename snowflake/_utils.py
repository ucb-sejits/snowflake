__author__ = 'nzhang-dev'


def make_enum(enum_name="", choices=()):
    properties = {}
    for i, choice in enumerate(choices):
        properties[str(choice).upper()] = property(lambda self, default=i: default)
    enum_type = type(enum_name, (), properties)
    enum_type.choices = property(lambda self: tuple(choices))
    return enum_type()