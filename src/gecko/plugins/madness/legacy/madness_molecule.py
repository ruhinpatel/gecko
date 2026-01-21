import copy
import json
from pathlib import Path


class geometry_parameters:
    def __init__(
        self,
        eprec=None,
        field=None,
        no_orient=None,
        psp_calc=None,
        pure_ae=None,
        symtol=None,
        core_type=None,
        units=None,
    ):

        self.eprec = 1e-4
        self.field = [0.0, 0.0, 0.0]
        self.no_orient = False
        self.psp_calc = False
        self.pure_ae = True
        self.symtol = -1e-2
        self.core_type = "none"
        self.units = "atomic"

        if eprec is not None:
            self.eprec = float(eprec)
        if field is not None:
            self.field = field
        if no_orient is not None:
            self.no_orient = no_orient
        if psp_calc is not None:
            self.psp_calc = psp_calc
        if pure_ae is not None:
            self.pure_ae = pure_ae
        if symtol is not None:
            self.symtol = symtol
        if core_type is not None:
            self.core_type = core_type
        if units is not None:
            self.units = units

    def __repr__(self):
        return f"eprec: {self.eprec}, field: {self.field}, no_orient: {self.no_orient}, psp_calc: {self.psp_calc}, pure_ae: {self.pure_ae}, symtol: {self.symtol}, core_type: {self.core_type}, units: {self.units}"

    def __to_json__(self):
        return {
            "eprec": float(self.eprec),
            "field": self.field,
            "no_orient": self.no_orient,
            "psp_calc": self.psp_calc,
            "pure_ae": self.pure_ae,
            "symtol": self.symtol,
            "core_type": self.core_type,
            "units": self.units,
        }


class MADMolecule:
    def __init__(
        self, orig=None, name=None, geometry=None, symbols=None, parameters=None
    ):
        if orig == None:
            self.geometry = []
            self.symbols = []
            self.parameters = geometry_parameters()
            self.name = name

            if geometry is not None:
                self.geometry = geometry
            if symbols is not None:
                self.symbols = symbols
            if parameters is not None:
                self.parameters = geometry_parameters(**parameters)
        else:
            self = copy.deepcopy(orig)
            self.parameters = copy.deepcopy(orig.parameters)

    def from_string(self, mol_file_string):
        for line in mol_file_string.splitlines():
            if line.startswith("end"):
                break
            if line and not line.startswith("#"):
                if line.startswith("geometry"):
                    continue
                split = line.split()
                if split[0] in self.parameters.__dict__.keys():
                    self.parameters.__dict__[split[0]] = split[1]

                else:
                    if len(split) != 4:
                        break
                    else:
                        self.symbols.append(split[0])
                        self.geometry.append([float(i) for i in split[1:]])

    def from_molfile(self, molfile: Path):
        self.parameters = geometry_parameters()
        self.geometry = []
        self.symbols = []
        with open(molfile) as f:
            for line in f:
                if line.startswith("end"):
                    break
                if line and not line.startswith("#"):
                    if line.startswith("geometry"):
                        continue
                    split = line.split()
                    if split[0] in self.parameters.__dict__.keys():
                        self.parameters.__dict__[split[0]] = split[1]

                    else:
                        if len(split) != 4:
                            break
                        else:
                            self.symbols.append(split[0])
                            self.geometry.append([float(i) for i in split[1:]])
        return self

    def to_molfile(self, molfile: Path):
        with open(molfile, "w") as f:
            f.write("geometry\n")
            # write parameters
            if self.parameters is not None:
                for key, value in self.parameters.__dict__.items():
                    f.write(f"  {key} {value}\n")
            for i, symbol in enumerate(self.symbols):
                f.write(
                    f"  {symbol} {self.geometry[i][0]} {self.geometry[i][1]} {self.geometry[i][2]}\n"
                )
            f.write("end\n")

    def add_atom(self, symbol, x, y, z):
        self.symbols.append(symbol)
        self.geometry.append([x, y, z])

    def to_json(self):
        return {
            "geometry": self.geometry,
            "symbols": self.symbols,
            "parameters": self.parameters.__to_json__(),
        }

    def __repr__(self):
        return f"parameters: {self.parameters}\ngeometry: {self.geometry}\nsymbols: {self.symbols}"


def dict_to_object(obj_class, data):
    return obj_class(**data)
