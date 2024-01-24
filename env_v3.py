from type_valuev3 import Type, Object
# The EnvironmentManager class keeps a mapping between each variable name (aka symbol)
# in a brewin program and the Value object, which stores a type, and a value.
class EnvironmentManager:
    def __init__(self):
        self.environment = [{}]

    # returns a VariableDef object
    def get(self, symbol):
        #print("env get symbol: ", symbol)
        if "." in symbol: # is an object
            obj_name = symbol.split(".")[0]
            obj_field = symbol.split(".")[1]
            value_obj = None
            for env in reversed(self.environment):
                if obj_name in env:
                    value_obj = env[obj_name]
                    break
            if value_obj != None:
                obj = value_obj.value()
                # HERE IS WHERE TO RECURSIVELY LOOK FOR PROTO STUFF
                return self.look_and_proto(obj, obj_field)
        else: # not an object
            for env in reversed(self.environment):
                if symbol in env:
                    return env[symbol]

        return None
    
    def look_and_proto(self, object, field):
        if field in object.members.keys():
            return object.members[field]
        if not "proto" in object.members.keys() or object.members["proto"].type() == Type.NIL:
            return None
        return self.look_and_proto(object.members["proto"].value(), field)

    def set(self, symbol, value, force_new_var_creation=False):
        if force_new_var_creation:
            self.environment[-1][symbol] = value
            return

        for env in reversed(self.environment):
            if symbol in env:
                env[symbol] = value
                return

        # symbol not found anywhere in the environment
        self.environment[-1][symbol] = value

    # create a new symbol in the top-most environment, regardless of whether that symbol exists
    # in a lower environment
    def create(self, symbol, value):
        self.environment[-1][symbol] = value

    # used when we enter a nested block to create a new environment for that block
    def push(self, env = None):
        if env is None:
            self.environment.append({})  # [{}] -> [{}, {}]
        else:
            self.environment.append(env)

    # used when we exit a nested block to discard the environment for that block
    def pop(self):
        self.environment.pop()

    def __enumerate(self):
        captured_so_far = set()
        for captured in reversed(self.environment):
            for var_name, value in captured.items():
                if var_name in captured_so_far:
                    continue
                captured_so_far.add(var_name)
                yield (var_name, value)

    def __iter__(self):
        return self.__enumerate()
