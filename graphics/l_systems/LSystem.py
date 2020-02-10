from typing import List
import numpy as np
import itertools
from math import cos, sin, pi


class LSystem:
    def __init__(self, axiom: List[str], constants: List[str], rules: dict, render_config, rules_probs=None):
        self.axiom = axiom
        self.constants = constants  # a variable list would be redundant (can get from rules dict)
        #  but is good to have constants as a separate list
        self.rules = rules
        self.rules_probs = rules_probs  # probabilities for stochastic L-System

        # Render specific config
        self.render_config = render_config
        self.azimuth_add = self.render_config['azimuth_add'] * (pi / 180)  # convert from degrees to radians
        self.inclination_add = self.render_config.get('inclination_add', 0) * (pi / 180)

        self.current_state = axiom

    def _apply_rule(self, val: str):
        # verify that given val is in the system alphabet
        if val not in self.rules and val not in self.constants:
            raise Exception("{} not in the alphabet".format(val))
        # if constant return list containing just the value itself
        elif val in self.constants:
            return [val]
        else:
            # if multiple rules for this value, means it is a stochastic L-System
            val_rules = self.rules[val]
            if isinstance(val_rules, list):
                # choose rules from the list based on given probabilities distribution
                choosen_rule = np.random.choice(val_rules, 1, p=self.rules_probs[val])
                return list(choosen_rule[0])
            else:
                return list(val_rules)

    def rewrite(self):
        new_state = list(itertools.chain.from_iterable([self._apply_rule(val) for val in self.current_state]))
        self.current_state = new_state


# Consider a turtle interpretation of a system state using cartesian coordinates
def render(state: List[str], draw_fun,
           azimuth_add, inclination_add=0.,
           azimuth=0., inclination=0.,
           pos=(0, 0, 0), line_length=1):

    while state:
        val = state.pop(0)
        # angles
        if val == '+':
            azimuth += azimuth_add
            inclination += inclination_add
        elif val == '-':
            azimuth -= azimuth_add
            inclination += inclination_add
        # move forward
        elif val == 'F' or val == 'G':
            new_pos = (
                pos[0] + line_length * cos(azimuth) * sin(inclination),
                pos[1] + line_length * sin(azimuth) * sin(inclination),
                pos[2] + line_length * cos(inclination)
            )
            draw_fun(pos, new_pos, {})
            pos = new_pos
        # branching
        elif val == '[':
            render(state=state, draw_fun=draw_fun, azimuth_add=azimuth_add, inclination_add=inclination_add,
                   azimuth=azimuth, inclination=inclination, pos=pos, line_length=line_length)
        elif val == ']':
            return
        # control growth (no drawing)
        elif val in ['X', 'Y']:
            continue
        else:
            raise Exception(f"No such value for rendering: {val}")


########################
# Common Examples
########################

dragon_curve = LSystem(
    axiom=['F', 'X'],
    constants=['-', '+', 'F'],
    rules={
        'X': 'X+YF+',
        'Y': '-FX-Y',
    },
    render_config={
        'azimuth_add': 90,
    }
)


koch_curve = LSystem(
    axiom=['F'],
    constants=['-', '+'],
    rules={
        'F': 'F+F-F-F+F',
    },
    render_config={
        'azimuth_add': 90,
    }
)


sierpinski_triangle = LSystem(
    axiom=['F', '-', 'G', '-', 'G'],
    constants=['-', '+'],
    rules={
        'F': 'F-G+F+G-F',
        'G': 'GG',
    },
    render_config={
        'azimuth_add': 120,
    }
)

sierpinski_arrowhead_curve = LSystem(
    axiom=['G'],
    constants=['-', '+'],
    rules={
        'G': 'F-G-F',
        'F': 'G+F+G',
    },
    render_config={
        'azimuth_add': 60,
    }
)

########################
# "Plants"
########################

fractal_plant = LSystem(
    axiom=['X'],
    constants=['-', '+', '[', ']'],
    rules={
        'X': 'F+[[X]-X]-F[-FX]+X',
        'F': 'FF',
    },
    render_config={
        'azimuth_add': 25,
    }
)

fractal_plant_b = LSystem(
    axiom=['F'],
    constants=['-', '+', '[', ']'],
    rules={
        'F': 'F[+F]F[-F][F]',
    },
    render_config={
        'azimuth_add': 20,
    }
)

fractal_plant_c = LSystem(
    axiom=['F'],
    constants=['-', '+', '[', ']'],
    rules={
        'F': 'FF-[-F+F+F]+[+F-F-F]',
    },
    render_config={
        'azimuth_add': 22.5,
    }
)

fractal_plant_e = LSystem(
    axiom=['X'],
    constants=['-', '+', '[', ']'],
    rules={
        'X': 'F[+X][-X]FX',
        'F': 'FF',
    },
    render_config={
        'azimuth_add': 25.7,
    }
)

stochastic_plant = LSystem(
    axiom=['F'],
    constants=['-', '+', '[', ']'],
    rules={
        'F': ['F[+F]F[-F]F', 'F[+F]F', 'F[-F]F'],
    },
    rules_probs={
        'F': [0.33, 0.33, 0.34],
    },
    render_config={
        'azimuth_add': 25,
    }
)
