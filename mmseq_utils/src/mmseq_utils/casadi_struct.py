from casadi.tools import entry, struct_symMX, struct_MX

def casadi_sym_struct_old(data: dict):
    param_entries = []
    for (name, shape) in data.items():
        param_entries.append(entry(name, shape=shape))
    
    return struct_symMX(param_entries)

def casadi_sym_struct(data: dict):
    param_entries = []
    for (name, expression) in data.items():
        param_entries.append(entry(name, expr=expression))
    
    return struct_MX(param_entries)

if __name__ == "__main__":
    import numpy as np
    import casadi as cs
    data = {"r_EEPos3": cs.MX.sym('r', 2), "W_EEPos3": cs.MX.sym('W',3,3)}
    p_struct = casadi_sym_struct(data)
    print(p_struct.getLabel)
    print(p_struct.labels)
    print(p_struct.keys())
    print(p_struct.order)


