import copy

from enum import Enum
from intbase import InterpreterBase


# Enumerated type for our different language data types
class Type(Enum):
    INT = 1
    BOOL = 2
    STRING = 3
    CLOSURE = 4
    NIL = 5
    OBJECT = 6


class Closure:
    def __init__(self, func_ast, env):
        #print("what is env? ", env)
        #print(" ")
        self.captured_env = copy.deepcopy(env)
        self.captured_env.environment = []
        for environment in env.environment:
            capture_dict = {}
            for symbol in environment:
                if environment[symbol].type() != Type.CLOSURE and environment[symbol].type() != Type.OBJECT:
                    capture_dict[symbol] = copy.deepcopy(environment[symbol])
                else:
                    capture_dict[symbol] = environment[symbol]
            self.captured_env.push(capture_dict)
        self.func_ast = func_ast
        self.type = Type.CLOSURE
        #print("original env: ", env.environment)
        #print(" ")
        #print("capture env: ", self.captured_env.environment)
        #print(" ")

class Object:
    def __init__(self, obj_ast):
        self.members = {}
        self.ast = obj_ast
    


# Represents a value, which has a type and its value
class Value:
    def __init__(self, t, v=None):
        self.t = t
        self.v = v

    def value(self):
        return self.v

    def type(self):
        return self.t

    def set(self, other):
        self.t = other.t
        self.v = other.v


def create_value(val):
    if val == InterpreterBase.TRUE_DEF:
        return Value(Type.BOOL, True)
    elif val == InterpreterBase.FALSE_DEF:
        return Value(Type.BOOL, False)
    elif isinstance(val, int):
        return Value(Type.INT, val)
    elif val == InterpreterBase.NIL_DEF:
        return Value(Type.NIL, None)
    elif isinstance(val, str):
        return Value(Type.STRING, val)
    else:
        raise ValueError("Unknown value type")


def get_printable(val):
    if val.type() == Type.INT:
        return str(val.value())
    if val.type() == Type.STRING:
        return val.value()
    if val.type() == Type.BOOL:
        if val.value() is True:
            return "true"
        return "false"
    return None