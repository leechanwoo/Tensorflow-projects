

class My_class(object):
    def __init__(self):
        self.m_integer = 10
        self.m_integer2 = 20

    def public_m(self):
        print(self.m_integer2)
        self.m_string = 'hello'

    def other_f(self):
        print(self.m_var)

    @classmethod
    def protected_m(cls):
        cls.m_var = 'var in protected'
        print(cls.m_integer)
