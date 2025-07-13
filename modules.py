# Sahar Jahani

import numpy as np
import sys
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fractions import Fraction
from enum import Enum
import hyperbola_equal_distance as hed

_var_names = ['p', 'q', 'r']
section_format = "\n{title}: \t{answer} "
comp_format= "\t{ind}- \t {comp}"

def initialize():
    global TESTS, PERCISION, PERCISION_DECIMAL, VERBOSE, OUTPUT, GIF, GIFNAME, ORIENTATION, OUTPUTFILE
    global SHOW_PLOT, SAVE_PLOT, SAVE_PLOT_NAME, ONLY_PLAYER, INPUT_FILE
    INPUT_FILE="game.stf"
    ONLY_PLAYER = None
    SHOW_PLOT = 1
    SAVE_PLOT = 0
    SAVE_PLOT_NAME = "plot.png"
    PERCISION_DECIMAL = 6
    PERCISION = 10**PERCISION_DECIMAL
    VERBOSE = 1
    GIF = 0
    GIFNAME = str(int(time.time()))+".gif"
    ORIENTATION = 1
    OUTPUT = ""
    OUTPUTFILE = "out.txt"
    TESTS = {
    0: np.array([[-3, 1, -3, 1], [2, 2, -2, -2], [-7, 1, 1, 1]]),
    1: np.array([[1, -2, 0, 1], [0, 0, -3, 1], [-2, 1, -2, 0]]),  # Selten's horse
    2: np.array([[-3, 1, -1, 2], [2, 1, -1, -3], [-2, 0, 0, 3]]),  # unique CME
    3: np.array([[3, 0, -1, 0], [3, -1, 0, 0], [-2, 0, 0, 3]]),  # Example 3 with a continuum mixed NE
    4: np.array([[3, 0, -1, 0], [3, -1, 0, 0], [0, 0, 0, 0]]),  # Extreme case with a player completely indifferent
    5: np.array([[7, -1, -1, -1], [-3, 1, -3, 1], [NFraction(1, 2), 0, NFraction(-1, 2), 0]]),
    6: np.array([[7, -1, -1, -1], [-1, -1, 0, 0], [NFraction(1, 2), 0, NFraction(-1, 2), 0]]),
    7: np.array([[-1, -2, -3, -4], [-2, -2, -2, -2], [-5, -3, -1, -1]]),
    8: np.array([[-4, -4, -4, -3], [0, 0, 1, 1], [NFraction(1, 2), 0, NFraction(-1, 2), 0]]),
    9: np.array([[4, -26, -16, 54], [1, 9, -9, -1], [23, -27, -27, 23]]),
    10: np.array([[1, 1, -9, 1], [-1, 9, -6, 4], [-23, +27, 27, -23]]),
    11: np.array([[-2, 1, -2, 1], [0, 3, 2, -2], [-2, -2, 1, 1]]),
    12: np.array([[-23, +27, 27, -23], [-23, +27, 27, -23], [-23, +27, 27, -23]]),
    13: np.array([[-23, +27, 27, -23], [2484, -2516, -2516, 2484], [-23, +27, 27, -23]]),
    14: np.array([[210000, -30000, -35000, 4979], [-10000, 2500, 2000, -499], [66, -22, -33, 5]]),
    15: np.array([[-23, +27, 27, -23], [-245, 255, 255, -245], [-23, +27, 27, -23]]),
    16: np.array([[2, -1, -4, 2], [2, -1, -4, 2], [2, -1, -4, 2]]),
    17: np.array([k_to_payoff([2, -3, 0, 0]), [-23, +27, 27, -23], [-23, +27, 27, -23]]),  # error resolved
    18: np.array([[2, 3, 4, 6], [-23, +27, 27, -23], [-23, +27, 27, -23]]),  # error resolved
    19: np.array([k_to_payoff([2, 2, -2, -1]), k_to_payoff([3, -2, 3, -1]), k_to_payoff([-100, 50, 50, -10])]),  # error resolved
    20: np.array([[-23, +75, 27, -23], [-23, +27, 27, -23], [-23, +21, 27, -23]]),  # 4 partial and one mixed equi total 9
    21: np.array([[-23, +100, 27, -23], [-23, +27, 27, -23], [-23, +21, 27, -23]]),
    22: np.array([k_to_payoff([16, -12, -4, 3]), k_to_payoff([16, -12, -4, 3]), k_to_payoff([16, -12, -4, 3])]),  # all deg hyp
    23: np.array([k_to_payoff([32, -23, -8, 6]), k_to_payoff([16, -12, -4, 3]), k_to_payoff([16, -12, -4, 3])]),
    24: np.array([k_to_payoff([32, -23, -8, 6]), k_to_payoff([16, -12, -4, 3]), k_to_payoff([32, -23, -8, 6])]),
    25: np.array([
        [1, NFraction(-1, 2), NFraction(-3, 4), NFraction(3, 8)],
        [1, NFraction(-1, 2), NFraction(-3, 4), NFraction(3, 8)],
        [1, NFraction(-1, 2), NFraction(-3, 4), NFraction(3, 8)]
    ]),
    26: roots2payoffs("1/2", "1/2", "1/2", "1/4", "1/4", "1/4"),  # all deg hyp
    27: roots_shift_payoffs("1/2", "1/2", "1/2", "51/100", "51/100", "51/100", 0, 0, "1/9999"),  # all deg hyp
    28: roots_shift_payoffs("1/2", "1/2", "1/2", "51/100", "51/100", "51/100", 0, 0, "1/999"),  # all deg hyp
    29: np.array([k_to_payoff([32, -23, -8, 6]), k_to_payoff([48, -35, -12, 9]), k_to_payoff([16, -12, -4, 3])]),  # one deg hyp and 2 hyp
    30: np.array([k_to_payoff([32, -23, -8, 6]),  k_to_payoff([16, -12, -4, 3]),
                 k_to_payoff([40, -10, -36, 9])]), # 30-error resolved
    
    31: np.array([k_to_payoff([16, -12, -4, 1]),  k_to_payoff([16, -12, -4, 2]), k_to_payoff([32, -24, -8, 7])]),
    32:roots_shift_payoffs("1/2", "2/3", "1/2", "2/3", "1/2", "2/3", "-1/36", "1/36",
                        "-1/36"),  # 32-error in mixed equilibria, NEW CASE
    33:np.array([k_to_payoff([32, -10, -8, 6]), k_to_payoff([16, -12, -4, 3]),  k_to_payoff([32, -23, -8, 6])]),
    34:np.array([k_to_payoff([32, -23, -8, 6]), k_to_payoff([32, -23, -8, 6]),  k_to_payoff([32, -23, -8, 6])]),
    
    35:np.array([k_to_payoff([12, -6, -4, 3]), k_to_payoff([0, 2, -1, 0]),  k_to_payoff([-4, 2, 2, -1])]),# 35- counter example for finite eq and degenerate game
    36:np.array([k_to_payoff([12, -6, -4, 3]), k_to_payoff([0, 2, -1, 0]),  k_to_payoff([-40, 20, 20, -11])]), # counter example for finite eq and degenerate game, perturb
    37:np.array([k_to_payoff([0, 0, 0, 0]), k_to_payoff([0, 0, 0, 0]),
                k_to_payoff([-40, 20, 20, -11])]),  # 37-degenerate 2D
    38:roots_shift_payoffs("1/2", "1/2", "1/2", "1/2", "1/3", "1/2", "0", "1/36", "1/36"),  # 38
    39:np.array([k_to_payoff([0, 0, 0, 0]), k_to_payoff([1, 2, 3, 4]),
                k_to_payoff([-40, 20, 20, -11])]),  # 39-degenerate 2D
    40:roots_shift_payoffs("1/5", "4/5", "2/5", "3/4", "1/2", "1/2", "1/100", "1/50", "1/100"), #40
    41: np.array([[-2, 1, -2, 1], [-3, -3, 1, 1], [-1, -1, 2, 2]]), # example of only 3 pute and partial equis
    42: np.array([[-2, 1, -2, 1], [-2, -3, -4, -1], [-1, -1, 2, 2]]),#quads=0 but no CME, IOS shows
    43: np.array([[-3, 1, -1, 2], [2, 1, -1, -3], [-4, -1, -1, 6]]),  # test2 changed to become nondeg with unique CME
    44: normalize_payoff(np.array([[[[0, 0, 0], [0, 2, 0]], [[-3, 0, 0], [1, -2, 0]]],
                                  [[[0, 0, -7], [0, 2, 1]], [[-3, 0, 1], [1, -2, 1]]]])) # a very degenerate case (was test0 before)
}



def process_file(file_path):
    try:

        # Open the file for reading
        with open(file_path, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()

            # Find the index of the line containing "Strategic Form:"
            strategic_form_index = next((i for i, line in enumerate(lines) if "Strategic Form:" in line), None)

            if strategic_form_index is not None:
                # Delete lines before "Strategic Form:"
                lines = lines[strategic_form_index + 1:]

                payoff = [[[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
                          [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]]
                for line in lines:
                    tmp = line.split(":")
                    inds = [int(i) for i in (tmp[0].strip()).split(" ")]
                    entries = [to_fraction(i) for i in (tmp[1].strip()).split(" ")]
                    if len(inds) == 3 and len(entries) == 3:
                        for i in range(3):
                            payoff[inds[0]][inds[1]][inds[2]][i] = entries[i]
                    else:
                        print("Error: can not read Strategic Form entries")
                        return None
                return np.array(payoff)
            else:
                print("Error: 'Strategic Form:' not found in the file.")
                return None
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


def to_fraction(s):
    if isinstance(s, str) and "." in s:
        s = float(s)
    if isinstance(s, float):
        num = int(abs(s)*PERCISION+0.5)  # round .5 away from zero
        if s < 0:
            num = -num
        return NFraction(num, PERCISION)
    # elif isinstance(s, Fraction):
    #     return NFraction(s.numerator, s.denominator)
    # any other s than a float or string containing '.':
    return NFraction(s)


def print_output():
    global OUTPUT, OUTPUTFILE
    if VERBOSE:
        print(OUTPUT)
    with open(OUTPUTFILE, 'w') as file:
        file. write(OUTPUT)


def normalize_payoff(payoff):
    if np.shape(payoff) == (2, 2, 2, 3):

        # in matrix representation, the order of indices is: [s1][s2][s3][0,1,2 for each player's payoff]
        t0 = [payoff[1][0][0][0]-payoff[0][0][0][0], payoff[1][1][0][0]-payoff[0][1][0][0],
              payoff[1][0][1][0]-payoff[0][0][1][0], payoff[1][1][1][0]-payoff[0][1][1][0]]

        t1 = [payoff[0][1][0][1]-payoff[0][0][0][1], payoff[0][1][1][1]-payoff[0][0][1][1],
              payoff[1][1][0][1]-payoff[1][0][0][1], payoff[1][1][1][1]-payoff[1][0][1][1]]

        t2 = [payoff[0][0][1][2]-payoff[0][0][0][2], payoff[1][0][1][2]-payoff[1][0][0][2],
              payoff[0][1][1][2]-payoff[0][1][0][2], payoff[1][1][1][2]-payoff[1][1][0][2]]

        return np.array([t0, t1, t2])
        # print('Normalized payoff: \n', self._normal_payoff_matrix)
    elif np.shape(payoff) == (3, 4):
        return payoff
        # print('Normalized payoff: \n', self._normal_payoff_matrix)
    else:
        raise ValueError('Error: Matrix dimensions error in normalization')


def k_to_payoff(MKLA):
    """
    turns k-coef (MKLA) of each player to the actual normalised matrix payoffs (ABCD)
    """
    if len(MKLA) != 4:
        print('just the k list for one player should be given')
        return None
    ABCD = [0, 0, 0, 0]

    ABCD[0] = MKLA[3]
    ABCD[1] = MKLA[1]+ABCD[0]
    ABCD[2] = MKLA[2]+ABCD[0]
    ABCD[3] = MKLA[0]-ABCD[0]+ABCD[1]+ABCD[2]

    return ABCD


def roots2payoffs(p1, q1, r1, p2, q2, r2):
    return np.array([
        k_to_payoff([1, -NFraction(r2), -NFraction(q1),
                    NFraction(r2)*NFraction(q1)]),
        k_to_payoff([1, -NFraction(p2), -NFraction(r1),
                    NFraction(p2)*NFraction(r1)]),
        k_to_payoff([1, -NFraction(q2), -NFraction(p1),
                    NFraction(q2)*NFraction(p1)])
    ])


def roots_shift_payoffs(p1, q1, r1, p2, q2, r2, k1, k2, k3):
    return np.array([
        k_to_payoff([1, -NFraction(r2), -NFraction(q1),
                    NFraction(r2)*NFraction(q1)+NFraction(k1)]),
        k_to_payoff([1, -NFraction(p2), -NFraction(r1),
                    NFraction(p2)*NFraction(r1)+NFraction(k2)]),
        k_to_payoff([1, -NFraction(q2), -NFraction(p1),
                    NFraction(q2)*NFraction(p1)+NFraction(k3)])
    ])


def payoff_to_k(ABCD):
    """
    turns k-coef (MKLA) of each player to the actual normalised matrix payoffs (ABCD)
    """
    if len(ABCD) != 4:
        print('just the k list for one player should be given')
        return None
    MKLA = [0, 0, 0, 0]

    MKLA[0] = ABCD[0]-ABCD[1]-ABCD[2]+ABCD[3]
    MKLA[1] = ABCD[1]-ABCD[0]
    MKLA[2] = ABCD[2]-ABCD[0]
    MKLA[3] = ABCD[0]

    return MKLA


def parse_arguments(args):
    payoff = None
    global VERBOSE, GIF, GIFNAME, ORIENTATION, OUTPUTFILE, SAVE_PLOT, SAVE_PLOT_NAME
    global INPUT_FILE, SHOW_PLOT, ONLY_PLAYER
    try:
        for i, arg in enumerate(args):
            if arg == "-n" or arg == "-k" or arg == "-r":
                # pay1 = list(map(NFraction, f"'{((args[i+1]).strip('[]').split(','))}'"))
                # pay2 = list(map(NFraction, f"'{((args[i+2]).strip('[]').split(','))}'"))
                # pay3 = list(map(NFraction, f"'{((args[i+3]).strip('[]').split(','))}'"))
                pay1 = list(map(NFraction, (args[i+1]).strip('[]').split(',')))
                pay2 = list(map(NFraction, (args[i+2]).strip('[]').split(',')))
                pay3 = list(map(NFraction, (args[i+3]).strip('[]').split(',')))

                if arg == "-n":
                    payoff = np.array([pay1, pay2, pay3])

                elif arg == "-k":
                    payoff = np.array([k_to_payoff(pay1), k_to_payoff(pay2), k_to_payoff(pay3)])

                elif arg == "-r":
                    payoff = roots_shift_payoffs(pay1[0], pay3[0], pay2[0], pay2[1],
                                                 pay1[1], pay3[1], pay1[2], pay2[2], pay3[2])
                    # ([p1,q2,K1], [r1,p2, K2] ,[q1, r2, K3] ) -> (p1,q1,r1,p2,q2,r2,k1,k2,k3)

            elif arg == "-t":
                payoff = TESTS[int(args[i+1])]
            elif arg == "-v":
                VERBOSE = int(args[i+1])
            elif arg == "-g":
                GIF = int(args[i+1])
                if (i+2) < len(args) and (not (args[i+2]).startswith("-")):
                    GIFNAME = args[i+2]+".gif"
            elif arg == "-o":
                ORIENTATION = int(args[i+1])
            elif arg == "-f":
                if (i+1) < len(args) and (not (args[i+1]).startswith("-")):
                    INPUT_FILE = args[i+1]
                raw_payoff = process_file(INPUT_FILE)
                payoff=normalize_payoff(raw_payoff)
            elif arg == "-w":
                if (i+1) < len(args) and (not (args[i+1]).startswith("-")):
                    OUTPUTFILE = args[i+1]+".txt"
            elif arg == "-p":
                SHOW_PLOT = int(args[i+1])
            elif arg == "-b":
                ONLY_PLAYER = int(args[i+1])-1
            elif arg == "-s" and int(args[i+1]):
                SAVE_PLOT = 1
                if (i+2) < len(args) and (not (args[i+2]).startswith("-")):
                    SAVE_PLOT_NAME = args[i+2]+".png"
            elif arg=="-q":
                sys.exit()

    except Exception as e:
        print("Error: arguments cannot be parsed!", e)

    return payoff


def prt(line):
    global OUTPUT
    OUTPUT += f"\n{line}"


def divide_in_fraction(num, denom):
    global PERCISION
    if denom == 0:
        if num == 0:
            return np.nan
        else:
            return np.inf

    elif isinstance(num, NFraction) and isinstance(denom, NFraction):
        return num/denom
    else:
        num = int(num*(PERCISION))
        denom = int(denom*(PERCISION))
        return NFraction(num, denom)


class Interval:
    def __init__(self, start, end):
        if start < end:
            self.s = start
            self.e = end
        else:
            self.s = end
            self.e = start

    def __repr__(self):
        if self.len() > 0:
            return f"[ {float(self.s):6f} , {float(self.e):6f} ]"
        else:
            return f"{{{float(self.s):6f}}}"

    def does_intersect(self, other):
        if self.s <= other.e and other.s <= self.e:
            return True
        return False

    def intersection(self, other):
        if self.does_intersect(other):
            vals = [self.s, self.e, other.s, other.e]
            vals.sort()
            return Interval(vals[1], vals[2])
        else:
            return None

    def len(self):
        return self.e-self.s

    def contains(self, val):
        if val >= self.s and val <= self.e:
            return True
        return False
# class Polygon:
#     def __init__(self,polygon_array):
#         self.polygon=polygon_array


class HypFraction:
    # y= (a x + b) / (c x + d)
    def __init__(self, a, b, c, d):
        self.ind = np.zeros(4)
        # if c == 0 and d == 0:
        #     raise ValueError
        # else:
        self.ind[0] = a
        self.ind[1] = b
        self.ind[2] = c
        self.ind[3] = d

    def get_val(self, x):
        num = (self.ind[0]*x+self.ind[1])
        denom = (self.ind[2]*x+self.ind[3])

        if denom == 0 and num == 0:
            return np.nan
        elif denom == 0:
            return np.inf
        else:
            return divide_in_fraction(num, denom)

    def is_equal(self, other):
        res = True
        c = None
        for i in range(4):
            if self.ind[i] == 0:
                res &= (self.ind[i] == other.ind[i])
            else:
                if c is None:
                    c = divide_in_fraction(other.ind[i], self.ind[i])
                else:
                    res &= (divide_in_fraction(other.ind[i], self.ind[i]) == c)

        return res

    def composition(self, other):
        # composition of fraction functions: res= self O other
        a = self.ind[0] * other.ind[0] + self.ind[1] * other.ind[2]
        b = self.ind[0] * other.ind[1] + self.ind[1] * other.ind[3]
        c = self.ind[2] * other.ind[0] + self.ind[3] * other.ind[2]
        d = self.ind[2] * other.ind[1] + self.ind[3] * other.ind[3]
        return HypFraction(a, b, c, d)

    def intersection(self, other):
        # returns answers of the quad: self=other
        global PERCISION_DECIMAL
        a = round((self.ind[0]*other.ind[2]-other.ind[0]*self.ind[2]), PERCISION_DECIMAL)
        b = round((self.ind[1]*other.ind[2]-other.ind[1]*self.ind[2] + self.ind[0]
                  * other.ind[3]-other.ind[0]*self.ind[3]), PERCISION_DECIMAL)
        c = round((self.ind[1]*other.ind[3]-other.ind[1]*self.ind[3]), PERCISION_DECIMAL)
        return compute_quadratic_roots(a, b, c)

    def switch_variable_and_interval(self, range):
        # y=frac(x), x in range => x=f(y), y in intvl
        f = None
        intvl = None
        if self.ind[2] == 0 and self.ind[0] == 0:
            # y=tmp
            tmp = divide_in_fraction(self.ind[1], self.ind[3])
            intvl = Interval(tmp, tmp)
        else:
            f = HypFraction(-self.ind[3], self.ind[1], self.ind[2], -self.ind[0])
            intvl = Interval(self.get_val(range.s), self.get_val(range.e))
        return f, intvl


class HypFracFunction:
    def __init__(self, input_ind, output_ind, fraction):
        self.input_ind = input_ind
        self.output_ind = output_ind
        self.fraction = fraction

    def __repr__(self):
        global PERCISION_DECIMAL
        inp = _var_names[self.input_ind]
        if (self.fraction.ind[0] != 0) or (self.fraction.ind[2] != 0):
            return f"{_var_names[self.output_ind]} = {repr_axb(self.fraction.ind[0],inp ,self.fraction.ind[1])} / {repr_axb(self.fraction.ind[2],inp ,self.fraction.ind[3])}"
        else:
            return f"{_var_names[self.output_ind]} = {float((self.fraction.ind[1])/ (self.fraction.ind[3])):+.{PERCISION_DECIMAL}f}"


class HypFracSegment:
    def __init__(self, p1FuncOfp2=None, p2Intervals=None, p2FuncOfp1=None, p1Intervals=None):
        self.p1Fp2 = p1FuncOfp2  # a HypFraction
        self.p2Intervals = p2Intervals
        self.p2Fp1 = p2FuncOfp1
        self.p1Intervals = p1Intervals


class IntersectionComponent:

    def __init__(self, basic_var, basic_var_interval):
        self.unres_vars = []
        self.basic_var = basic_var
        self.basic_var_interval = basic_var_interval

        # for all the cases, input_ind=basic_var
        self.functions = []

    def __repr__(self):
        global PERCISION_DECIMAL
        res = ""
        if self.basic_var_interval.len() > 0:
            res += f"\n\t {_var_names[self.basic_var]} in {self.basic_var_interval} "
            for f in self.functions:
                res += f"\n\t {f}"
        else:
            point = [0, 0, 0]
            point[self.basic_var] = float(self.basic_var_interval.s)
            for f in self.functions:
                if f.input_ind == self.basic_var:
                    point[f.output_ind] = float(f.fraction.get_val(point[self.basic_var]))
            for var in self.unres_vars:
                point[var] = _var_names[var]
            
            repre=["","",""]
            for i in range(3):
                if isinstance(point[i],str):
                    repre[i]= point[i]
                else:
                    repre[i]=f"{point[i]:.{PERCISION_DECIMAL}f}"
                
            res += f"({repre[0]}, {repre[1]}, {repre[2]})"

        for var in self.unres_vars:
            if var != self.basic_var:
                res += f"\n\t {_var_names[var]} in {Interval(0,1)} unrestricted"
        return res

    def get_range(self):
        rng = [None, None, None]
        rng[self.basic_var] = self.basic_var_interval
        for i in self.unres_vars:
            rng[i] = Interval(0, 1)
        for f in self.functions:
            if rng[f.input_ind] is not None:
                inp = rng[f.input_ind]
                rng[f.output_ind] = Interval(f.fraction.get_val(inp.s), f.fraction.get_val(inp.e))
        return rng

    # def contains(self, other):
        

class NFraction(Fraction):
    
    def __repr__(self):
        if self.denominator != 1:
            return f"{self.numerator}/{self.denominator}"
        else:
            return f"{self.numerator}"


def compute_quadratic_roots(a, b, c):
    """
    a x^2 + b x + c = 0
    returns None if infinite solutions. otherwise, a list of roots
    """
    ans = []
    res = []
    if a == 0:
        if b == 0:
            if c == 0:
                # 0-1 segment
                ans = None
                res = None
        else:
            ans.append((-c/b))
    else:
        dis_form = b * b - 4 * a * c
        sqrt_val = math.sqrt(abs(dis_form))

        if dis_form > 0:
            sol1 = (-b + sqrt_val) / (2 * a)
            sol2 = (-b - sqrt_val) / (2 * a)
            ans = [sol1, sol2]

        elif dis_form == 0:
            sol1 = (-b / (2 * a))
            ans = [sol1]
    if ans is not None:
        for sol in ans:
            if is_in_range(sol):         
                res.append(sol)
    return res


def repr_axb(a, x, b):
    # represent a x +b
    global PERCISION_DECIMAL
    output = ""
    if (a != 0) or (b != 0):
        output += f"({float(a):+.{PERCISION_DECIMAL}f} {x}" if (a != 0) else "("
        output += f"{float(b):+.{PERCISION_DECIMAL}f})" if (b != 0) else ")"
        # fraction presentation but it looks messy
        # output += f"(({to_fraction(a)}) {x}" if (a != 0) else "("
        # output += f"({to_fraction(b)}))" if (b != 0) else ")"
    else:
        output += "0"
    return output


def compute_intersection_of_BRs(BR0, BR1):
    res = []
    if len(BR0) == 0 or len(BR1) == 0:
        # at least one of them is all the square
        BR = BR0+BR1
        for i in range(len(BR)-1):
            res.append([BR[i], BR[i+1]])
    else:
        for i in range(len(BR0)-1):
            for j in range(len(BR1)-1):
                part0 = [BR0[i], BR0[i+1]]
                part1 = [BR1[j], BR1[j+1]]
                # print('part0: ',part0)
                # print('part1: ',part1)
                res += intersection_of_segments(
                    part0, part1)

    return res


def intersection_of_segments(part0, part1):
    res = []
    # if any of these segments is a point
    if part0[0] == part0[1] or part1[0] == part1[1]:
        if (part0[0] == part0[1]) and (is_between_values(part0[0][0], part1[0][0], part1[1][0])) and (is_between_values(part0[0][1], part1[0][1], part1[1][1])):
            res.append([part0[0]])
        elif (part1[0] == part1[1]) and (is_between_values(part1[0][0], part0[0][0], part0[1][0])) and (is_between_values(part1[0][1], part0[0][1], part0[1][1])):
            res.append([part1[0]])
    else:

        # [dy0,dx0][dy1,dx1]
        delta = [[(part0[0][1]-part0[1][1]), (part0[0][0]-part0[1][0])],
                 [(part1[0][1]-part1[1][1]), (part1[0][0]-part1[1][0])]]
        # print('delta:',delta)

        # both lines y=sth
        if ((delta[0][0] == delta[1][0]) and (delta[0][0] == 0) and (part0[0][1] == part1[0][1])):
            # if they have an intersection
            if (min(part0[0][0], part0[1][0]) <= max(part1[0][0], part1[1][0])) and (max(part0[0][0], part0[1][0]) >= min(part1[0][0], part1[1][0])):
                temp = [part0[0][0], part0[1][0], part1[0][0], part1[1][0]]
                temp.sort()
                if temp[1] != temp[2]:
                    res += [[[temp[1], part0[0][1]], [temp[2], part0[0][1]]]]
                else:
                    res += [[[temp[1], part0[0][1]]]]

        # both lines x=sth
        elif ((delta[0][1] == delta[1][1]) and (delta[0][1] == 0) and (part0[0][0] == part1[0][0])):
            # if they have an intersection
            if (min(part0[0][1], part0[1][1]) <= max(part1[0][1], part1[1][1])) and (max(part0[0][1], part0[1][1]) >= min(part1[0][1], part1[1][1])):
                temp = [part0[0][1], part0[1][1], part1[0][1], part1[1][1]]
                temp.sort()
                if temp[1] != temp[2]:
                    res += [[[part0[0][0], temp[1]], [part0[0][0], temp[2]]]]
                else:
                    res += [[[part0[0][1], temp[1]]]]

        # prependicular segments
            # first one y=sth second one x=sth
        elif (delta[0][1] == 0) and (delta[1][0] == 0):
            point = [part0[0][0], part1[0][1]]
            # print(point)
            if (is_between_values(point[1], part0[0][1], part0[1][1])) and (is_between_values(point[0], part1[0][0], part1[1][0])):
                res += [[point]]
        elif (delta[0][0] == 0) and (delta[1][1] == 0):
            point = [part1[0][0], part0[0][1]]
            if (is_between_values(point[0], part0[0][0], part0[1][0])) and (is_between_values(point[1], part1[0][1], part1[1][1])):
                res += [[point]]

    return res


# it's for 2x2 games
def compute_best_responce(player, A, B):
    # player means in this 2*2 game, it's row player (0) or column player (1)
    result = []
    if A == B:
        if A == 0:
            # all the square
            pass
        elif A > 0:
            # p=1
            result.append([1, 0])
            result.append([1, 1])
        else:
            # p=0
            result.append([0, 0])
            result.append([0, 1])
    else:
        mid = NFraction(A, A-B)
        if A > B:
            if mid >= 0 and mid <= 1:
                result.append([0, 1])
                result.append([0, mid])
                result.append([1, mid])
                result.append([1, 0])
            elif mid > 1:
                result.append([1, 1])
                result.append([1, 0])
            elif mid < 0:
                result.append([0, 1])
                result.append([0, 0])
        elif B > A:
            if mid >= 0 and mid <= 1:
                result.append([0, 0])
                result.append([0, mid])
                result.append([1, mid])
                result.append([1, 1])
            elif mid > 1:
                result.append([0, 0])
                result.append([0, 1])
            elif mid < 0:
                result.append([1, 0])
                result.append([1, 1])

    if player == 0:
        return result
    elif player == 1:
        for point in result:
            temp = point[0]
            point[0] = point[1]
            point[1] = temp
        return result


def validate_NE_point(point):
    if (point[0] >= 0 and point[0] <= 1) and (point[1] >= 0 and point[1] <= 1):
        return [point]
    else:
        return None


def is_in_cube(point):
    for i in range(len(point)):
        if point[i] < 0 or point[i] > 1:
            return False
    return True


def validate_NE_segment(point0, point1):
    min = point0
    max = point1

    if (point_value(min) > point_value(max)):
        min = point1
        max = point0

    if point0 == point1:
        return validate_NE_point(point0)

    vmin = validate_NE_point(min)
    vmax = validate_NE_point(max)

    line_min = []
    line_max = []

    # x=sth
    if min[0] == max[0]:
        if min[0] >= 0 and min[0] <= 1:
            line_min = [min[0], 0]
            line_max = [min[0], 1]
        else:
            return None
    # y=sth
    elif min[1] == max[1]:
        if min[1] >= 0 and min[1] <= 1:
            line_min = [0, min[1]]
            line_max = [1, min[1]]
        else:
            return None
    else:
        raise ValueError("lines are neither horizontal nor vertical")

    res = []

    if (vmin is not None) and (vmax is not None):
        res = [min, max]
    elif (vmin is None) and (vmax is not None):
        res = [line_min, vmax]
    elif (vmin is not None) and (vmax is None):
        res = [vmin, line_max]
    else:
        # both points are outside the cube, on the same side
        if (point_value(min) < point_value(line_min) and point_value(max) < point_value(line_min)
                ) or (point_value(min) > point_value(line_max) and point_value(max) > point_value(line_max)):
            return None
        # both points outside, on different sides
        else:
            res = [line_min, line_max]

    if res[0] == res[1]:
        return validate_NE_point(res[0])

    return res


def point_value(point):
    return math.sqrt(point[0]**2+point[1]**2)


def is_between_values(cand, val0, val1):
    return cand >= min(val0, val1) and cand <= max(val0, val1)


def add_uniquly(lst, cand):
    # attention: each point has 3 coordinates here
    added = False
    pop_list = []
    rng = range(len(lst))
    for item in lst:
        if len(item) == 1:
            if len(cand) == 1:
                if item == cand:
                    added = True
                    break
            elif len(cand) == 2:
                if segment_contains_point(cand, item[0]):
                    pop_list.append(item)

        elif len(item) == 2:
            if len(cand) == 1:
                if segment_contains_point(item, cand[0]):
                    added = True
                    break
            elif len(cand) == 2:
                dif_cand = [j for j, val in enumerate(
                    (np.array(cand[0]) == np.array(cand[1]))) if not val]
                dif_item = [j for j, val in enumerate(
                    (np.array(item[0]) == np.array(item[1]))) if not val]
                # print("diff_cand", dif_cand)
                # print("dif_item", dif_item)
                # equal slope
                if dif_cand == dif_item and len(dif_cand) == 1:
                    eql_ind = list(set([0, 1, 2]) - set(dif_cand))
                    same_line = True
                    for t in eql_ind:
                        same_line &= (cand[0][t] == item[0][t])
                    if same_line:
                        # changing axis
                        ax = dif_cand[0]
                        # check if they intersect
                        if (min(cand[0][ax], cand[1][ax]) <= max(item[0][ax], item[1][ax])) and (max(cand[0][ax], cand[1][ax]) >= min(item[0][ax], item[1][ax])):
                            tmp = [cand[0][ax], cand[1][ax],
                                   item[0][ax], item[1][ax]]
                            tmp.sort()
                            item[0][ax] = tmp[0]
                            item[1][ax] = tmp[3]
                            added = True
                            break

    if not added:
        lst.append(cand)
    if len(pop_list) > 0:
        for item in pop_list:
            lst.remove(item)


def segment_contains_point(segment, point):
    head = segment[0]
    tail = segment[1]
    res = True

    dif_axis_segment = [i for i, val in enumerate(
        (np.array(head) == np.array(tail))) if not val]

    if len(dif_axis_segment) == 1:
        for i in range(3):
            res &= (is_between_values(point[i], head[i], tail[i]))

    else:
        res = False

    return res


# def divide_in_fraction(num, denom):
#     num = int(num*(md.PERCISION))
#     denom = int(denom*(md.PERCISION))
#     return Fraction(num, denom)


def get_BR_function_segments(coef):  # interval in x direction or y
    shape, _, surf_points, _ = compute_shape_and_points(coef)
    surf_n = len(surf_points)
    ch_ax = 0
    segments = []

    if shape == BR_Shape.Hyperbola:
        p1Intvl = []
        p2Intvl = []
        center = [divide_in_fraction(-coef[2], coef[0]),
                  divide_in_fraction(-coef[1], coef[0])]

        # ASC sort based on ch_ax
        for k in range(surf_n):
            for l in range(k+1, surf_n):
                if surf_points[k][ch_ax] > surf_points[l][ch_ax]:
                    tmp = surf_points[k]
                    surf_points[k] = surf_points[l]
                    surf_points[l] = tmp
        if surf_n == 4:
            p1Intvl = [Interval(surf_points[0][ch_ax], surf_points[1][ch_ax]), Interval(
                surf_points[2][ch_ax], surf_points[3][ch_ax])]
            p2Intvl = [Interval(surf_points[0][1-ch_ax], surf_points[1][1-ch_ax]),
                       Interval(surf_points[2][1-ch_ax], surf_points[3][1-ch_ax])]

        elif surf_n == 3:

            if surf_points[1][ch_ax] < center[ch_ax]:
                p1Intvl = [Interval(surf_points[0][ch_ax],
                                    surf_points[1][ch_ax])]
                p2Intvl = [Interval(surf_points[0][1-ch_ax],
                                    surf_points[1][1-ch_ax])]
                # we don't consider surf_point[2] because it will be considered in the partially mixed equilibria
            else:
                p1Intvl = [Interval(surf_points[1][ch_ax],
                                    surf_points[2][ch_ax])]
                p2Intvl = [Interval(surf_points[1][1-ch_ax],
                                    surf_points[2][1-ch_ax])]
        elif surf_n == 2:
            if (surf_points[1][ch_ax] - center[ch_ax]) * (surf_points[0][ch_ax] - center[ch_ax]) > 0:
                p1Intvl = [Interval(surf_points[0][ch_ax],
                                    surf_points[1][ch_ax])]
                p2Intvl = [Interval(surf_points[0][1-ch_ax],
                                    surf_points[1][1-ch_ax])]

        f1 = HypFraction(-coef[2], -coef[3], coef[0], coef[1])
        f2 = HypFraction(-coef[1], -coef[3], coef[0], coef[2])

        segments.append(HypFracSegment(f1, p2Intvl, f2, p1Intvl))
    elif shape == BR_Shape.Line or shape == BR_Shape.DegHyperbola:
        cs = degHyperbola_to_lines(shape, coef)
        for c in cs:
            f1 = HypFraction(-c[2], -c[3], c[0], c[1])
            f2 = HypFraction(-c[1], -c[3], c[0], c[2])
            p1Intvl = []
            p2Intvl = []
            if c[1] == 0:
                f1 = None
                p1Intvl = [Interval(0, 1)]
                tmp = divide_in_fraction(-c[3], c[2])
                p2Intvl = [Interval(tmp, tmp)]
            elif c[2] == 0:
                f2 = None
                p2Intvl = [Interval(0, 1)]
                tmp = divide_in_fraction(-c[3], c[1])
                p1Intvl = [Interval(tmp, tmp)]
            else:
                if surf_n == 2:
                    p1Intvl = [Interval(surf_points[0][0], surf_points[1][0])]
                    p2Intvl = [Interval(surf_points[0][1], surf_points[1][1])]
                else:
                    f1 = f2 = None

            segments.append(HypFracSegment(f1, p2Intvl, f2, p1Intvl))

    return segments


def intersection_of_intervals(set1, set2):
    collide = []
    for part1 in set1:
        for part2 in set2:
            if len(part1) == 1 and len(part2) == 1 and part1[0] == part2[0]:
                collide.append(part1)
            elif len(part1) == 1 and part2[0] <= part1[0] and part2[1] >= part1[0]:
                collide.append(part1)
            elif len(part2) == 1 and part1[0] <= part2[0] and part1[1] >= part2[0]:
                collide.append(part2)
            else:
                if (part1[0] <= part2[1] and part2[0] <= part1[1]):
                    new_part = [0, 0]
                    new_part[0] = max(part1[0], part2[0])
                    new_part[1] = min(part1[1], part2[1])
                    if new_part[0] == new_part[1]:
                        collide.append([new_part[0]])
                    else:
                        collide.append(new_part)

    return collide


def get_cnew_and_shape(c1, c2):

    c_new = [0, 0, 0, 0]
    c_new[0] = (c1[2]*c2[0])-(c1[0]*c2[1])
    c_new[1] = (c1[2]*c2[2])-(c1[0]*c2[3])
    c_new[2] = (c1[3]*c2[0])-(c1[1]*c2[1])
    c_new[3] = (c1[3]*c2[2])-(c1[1]*c2[3])
    shape_new, _, surf_new_points, _ = compute_shape_and_points(c_new)

    return c_new, shape_new, surf_new_points


def degHyperbola_to_lines(shape, c):
    if (shape == None and c[0] != 0 and c[1]*c[2]-c[0]*c[3] == 0) or shape == BR_Shape.DegHyperbola:
        c1 = [0, c[0], 0, c[2]]
        c2 = [0, 0, c[0], c[1]]
        return [c1, c2]
    else:
        return [c]


def intersection_of_BR_lines(c1, c2):
    res = None
    c_new = [0, 0, 0, 0]
    c_new[0] = (c1[2]*c2[0])-(c1[0]*c2[1])
    c_new[1] = (c1[2]*c2[2])-(c1[0]*c2[3])
    c_new[2] = (c1[3]*c2[0])-(c1[1]*c2[1])
    c_new[3] = (c1[3]*c2[2])-(c1[1]*c2[3])
    shape_new, _, surf_new_points, _ = compute_shape_and_points(c_new)

    if np.array_equal(np.array(c_new), np.zeros(4)):
        # a line or a surface
        if c1[1] == 0 and c2[2] == 0:
            var0 = divide_in_fraction(-c1[3], c1[2])
            var1 = divide_in_fraction(-c2[3], c2[1])
            surf_new_points = [[var0, var1, 0], [var0, var1, 1]]
            res = surf_new_points
        elif c1[1] != 0 and c2[2] != 0:
            # all square
            if c1[1] != 0:
                vars = [-c1[2], -c1[3], 0, c1[1]]
                var_i = 0
            else:
                vars = [-c2[1], -c2[3], 0, c2[2]]
                var_i = 1
            corners = [[0, 0], [0, 1], [1, 1], [1, 0]]
            for point in corners:
                point.append(fraction_for_val(point[var_i], vars))
            res = corners
        else:
            print("Error: intersection of 2 brs, line-line- cnew=0")

    else:
        # constant means no intersection
        if shape_new == BR_Shape.Line:
            for point in surf_new_points:
                point[2] = (fraction_for_val(point[0], [-c1[2], -c1[3], 0, c1[1]])) if (not math.isnan(
                    fraction_for_val(point[0], [-c1[2], -c1[3], 0, c1[1]])))else (fraction_for_val(point[1], [-c2[1], -c2[3], 0, c2[2]]))
            res = surf_new_points
        else:
            print("Error: intersection of 2 brs, line-line- cnew!=0")
    return res


def switch_fraction_variable_and_interval(frac, range):
    # y=frac(x), x in range => x=f(y), y in intvl
    f = None
    intvl = None
    if frac.ind[2] == 0 and frac.ind[0] == 0:
        # y=tmp
        tmp = divide_in_fraction(frac.ind[1], frac.ind[3])
        intvl = Interval(tmp, tmp)
    else:
        f = HypFraction(-frac.ind[3], frac.ind[1], frac.ind[2], -frac.ind[0])
        intvl = Interval(frac.get_val(range.s), frac.get_val(range.e))
    return f, intvl


def compute_shape_and_points(coef):
    # ---1---
    # |       |
    # 0       2
    # |       |
    # ---3---
    cri_points = [-1, -1, -1, -1]
    corners = [[0, 0], [0, 1], [1, 1], [1, 0]]
    corners_val = -1*np.ones(4)

    shape = None

    # later in code, for segments it should be changed, we don't have player here
    P1 = 1
    P2 = 2

    for i in range(len(corners)):
        point = corners[i]
        val = coef[0]*point[0]*point[1] + coef[1] * \
            point[0]+coef[2]*point[1]+coef[3]
        corners_val[i] = map_to_what(val)

    if coef[0] == 0:
        if coef[1] == 0:
            if coef[2] == 0:
                shape = BR_Shape.Constant
            else:
                shape = BR_Shape.Line
                tmp = divide_in_fraction(-coef[3], coef[2])
                if is_in_interior(tmp):
                    cri_points[0] = cri_points[2] = tmp

        else:
            shape = BR_Shape.Line
            tmp3 = divide_in_fraction((-coef[3]), coef[1])
            tmp1 = divide_in_fraction((-coef[3]-coef[2]), coef[1])
            if is_in_interior(tmp3):
                cri_points[3] = tmp3
            if is_in_interior(tmp1):
                cri_points[1] = tmp1

            if coef[2] != 0:
                tmp0 = divide_in_fraction((-coef[3]), coef[2])
                tmp2 = divide_in_fraction((-coef[3]-coef[1]), coef[2])
                if is_in_interior(tmp0):
                    cri_points[0] = tmp0
                if is_in_interior(tmp2):
                    cri_points[2] = tmp2

    else:
        # BC-AD=0
        if coef[1]*coef[2]-coef[0]*coef[3] == 0:
            shape = BR_Shape.DegHyperbola
            q = divide_in_fraction((-coef[2]), coef[0])
            r = divide_in_fraction((-coef[1]), coef[0])

            if is_in_interior(q):
                cri_points[1] = q
                cri_points[3] = q

            if is_in_interior(r):
                cri_points[0] = r
                cri_points[2] = r

        else:
            shape = BR_Shape.Hyperbola
            x0 = divide_in_fraction(-coef[3], coef[1])
            x1 = divide_in_fraction(-coef[3]-coef[2], coef[1]+coef[0])
            y0 = divide_in_fraction(-coef[3], coef[2])
            y1 = divide_in_fraction(-coef[3]-coef[1], coef[2]+coef[0])
            if is_in_interior(x0):
                cri_points[3] = x0
            if is_in_interior(x1):
                cri_points[1] = x1
            if is_in_interior(y0):
                cri_points[0] = y0
            if is_in_interior(y1):
                cri_points[2] = y1

    all_points = []
    surf_points = []
    surf_points_index = []
    for i in range(len(corners)):
        corners[i].append(corners_val[i])
        all_points.append(corners[i].copy())
        if math.isnan(corners[i][2]):
            surf_points.append(corners[i].copy())
            surf_points_index.append(len(all_points)-1)
        if cri_points[i] != -1:
            point = cri_point_coordinate(i, cri_points[i])
            all_points.append(point)
            surf_points_index.append(len(all_points)-1)
            surf_points.append(point.copy())

    return shape, all_points, surf_points, surf_points_index


def create_hyperbola_small_surfaces(points, player):
    all = []
    for i in range(len(points)-1):
        all. append(modify_points_ax(
            define_BR_surface([points[i], points[i+1]]), player))
    return all


def prepare_mapped_surface(points, player):
    return modify_points_ax(set_nan_values(points), player)


def prepre_BR_main(points, player):
    return modify_points_ax(define_BR_surface(points), player)


def is_pure_strategy(point, length=None):
    res = True
    if length is None:
        length = len(point)
    for i in range(length):
        res &= (point[i] == 0 or point[i] == 1)
    return res


def get_equal_distance_hyperbola_points(c, start, end, n, axis=0):
    # if we have x and we want y, axis=0. the other way, axis=1
    # start and end determine the interval of axis
    if end <= start or n <= 1:
        res = [[start, get_point_on_hyperbola(c, start, axis), np.nan]]
    else:
        if axis == 0:
            res = hed.dots(
                start, end, n, lambda x: (-c[1]*x-c[3])/(c[0]*x+c[2]))

        else:
            res = hed.dots(
                start, end, n, lambda y: (-c[2]*y-c[3])/(c[0]*y+c[1]))
            for point in res:
                tmp = point[0]
                point[0] = point[1]
                point[1] = tmp
        for point in res:
            point.append(np.nan)
    return res


def get_point_on_hyperbola(c, val, axis=0):
    # if we have x and we want y, axis=0. the other way, axis=1
    if axis == 0:
        num = -c[3]-c[1]*val
        denom = c[2]+c[0]*val
    else:
        num = -c[3]-c[2]*val
        denom = c[1]+c[0]*val
    return divide_in_fraction(num, denom)


def same_arc_of_hyperbola(c, p1, p2):
    if (c[0] != 0 and (c[1]*c[2]-c[0]*c[3]) != 0):
        center = [divide_in_fraction(-c[2], c[0]),
                  divide_in_fraction(-c[1], c[0])]
        tmp = (center[0]-p1[0])*(center[0]-p2[0])
        if tmp > 0:
            return True
        elif tmp < 0:
            return False
        else:
            print("same_arc_of_hyperbola Error: one of the points is center")
            return None

    else:
        print("same_arc_of_hyperbola Error: not a hyperbola", c)
        return None


def define_BR_surface(points, free_ind=2):
    n = len(points)
    surf = np.zeros((2*n, 3))
    for i in range(n):
        for j in range(3):
            surf[i][j] = points[i][j]
        surf[i][free_ind] = 0
    for i in range(n-1, -1, -1):
        for j in range(3):
            surf[2*n-i-1][j] = points[i][j]
        surf[2*n-i-1][free_ind] = 1
    return surf
# def define_BR_surface(points, free_ind=2):##############################################################

#     surf = []
#     for i in range(len(points)):
#         tmp=[0,0,0]
#         for j in range(3):
#             tmp[j] = points[i][j]
#         tmp[free_ind] = 0
#         if is_in_cube(tmp):
#             surf.append(tmp)
#     n = len(surf)
#     for i in range(n-1, -1, -1):
#         tmp=[0,0,0]
#         for j in range(3):
#             tmp[j] = surf[i][j]
#         tmp[free_ind] = 1
#         surf.append(tmp)
#     return np.array(surf)
    # for i in range(n):
    #     surf[i][0] = points[i][0]
    #     surf[i][1] = points[i][1]
    # for i in range(n-1, -1, -1):
    #     surf[2*n-i-1][0] = points[i][0]
    #     surf[2*n-i-1][1] = points[i][1]
    #     surf[2*n-i-1][2] = 1
    # return surf


def set_nan_values(points):
    value = -1
    points_new = []
    for p in points:
        p_new = p.copy()
        if not math.isnan(p[2]):
            value = p[2]
            break
    if value != -1:
        for p in points:
            p_new = p.copy()
            p_new[2] = value
            points_new.append(p_new)
        return points_new
    else:
        print("Error: setting nan values for", points_new)
        return []


def modify_points_ax(points, player):
    new_points = np.zeros(np.shape(points))
    for i in range(len(points)):
        new_points[i][(player+1) % 3] = points[i][0]
        new_points[i][(player-1) % 3] = points[i][1]
        new_points[i][player] = points[i][2]

        # new axis to match matrix representation and cube
    change_representation_xyz_to_qrp(new_points)

    return new_points


def change_representation_xyz_to_qrp(lst):

    for comp in lst:
        points = []
        if len(comp) == 2:
            points.append(comp[0])
            points.append(comp[1])
        else:
            points.append(comp)

        for point in points:
            p = point[0]
            q = point[1]
            r = point[2]

            point[0] = q
            point[1] = r
            point[2] = p


def cri_point_coordinate(index, val):
    #       1
    #   ---------
    #   |       |
    # 0 |       | 2
    #   |       |
    #   ---------
    #       3
    res = []
    if index == 0:
        res = [0, val]
    elif index == 1:
        res = [val, 1]
    elif index == 2:
        res = [1, val]
    elif index == 3:
        res = [val, 0]
    else:
        return None
    res.append(np.nan)
    return res


def map_to_what(val):
    if val > 0:
        return 1
    elif val < 0:
        return 0
    else:
        return None


def is_on_BR(coef, x, y):
    # if res=0 it is on BR, -1 for for being less, +1 for being more
    val = coef[0]*x*y+coef[1]*x+coef[2]*y+coef[3]
    if val > 0:
        return 1
    elif val == 0:
        return 0
    else:
        return -1


def is_on_BR_TF(coef, x, y):
    val = coef[0]*x*y+coef[1]*x+coef[2]*y+coef[3]
    if abs(val) < computation_accuracy:
        return True
    else:
        return False


def is_in_range(x, y=None):
    val_x = (x >= 0 and x <= 1) and (
        not (math.isnan(x)) and (not (math.isinf(x))))
    if y == None:
        return val_x
    else:
        return val_x and (y >= 0 and y <= 1) and (not (math.isnan(y)) and (not (math.isinf(y))))


def is_in_interior(x, y=None, z=None):
    val_x = (x > 0 and x < 1) and (not (math.isnan(x)) and (not (math.isinf(x))))
    if y == None:
        return val_x
    else:
        val_y = val_x and (y > 0 and y < 1) and (not (math.isnan(y)) and (not (math.isinf(y))))
        if z == None:
            return val_y
        else:
            return val_y and (z > 0 and z < 1) and (not (math.isnan(z)) and (not (math.isinf(z))))


def get_critical_points(coef):
    # critical points on 4 sides of the square [x=0, x=1][y=0, y=1]
    # -1 means all the line
    points = np.zeros((2, 2))

    # x=0
    tmp = None
    if coef[2] != 0 and is_in_range(NFraction(-coef[3], coef[2])):
        tmp = NFraction(-coef[3], coef[2])
    elif coef[2] == 0 and (coef[0] != 0 or coef[1] != 0) and coef[3] == 0:
        tmp = -1
    points[0][0] == tmp

    # x=1
    tmp = None
    if coef[2]+coef[0] != 0 and is_in_range(NFraction(-coef[3]-coef[1], coef[2]+coef[0])):
        tmp = NFraction(-coef[3]-coef[1], coef[2]+coef[0])
    elif coef[2] == -coef[0] and coef[3] == 0:
        tmp = -1
    points[0][1] == tmp

    # y=0
    tmp = None
    if coef[1] != 0 and is_in_range(NFraction(-coef[3], coef[1])):
        tmp = NFraction(-coef[3], coef[1])
    elif coef[1] == 0 and (coef[0] != 0 or coef[2] != 0) and coef[3] == 0:
        tmp = -1
    points[1][0] == tmp

    # y=1
    tmp = None
    if coef[1]+coef[0] != 0 and is_in_range(NFraction(-coef[3]-coef[2], coef[1]+coef[0])):
        tmp = NFraction(-coef[3]-coef[2], coef[1]+coef[0])
    elif coef[1] == -coef[0] and coef[3] == 0:
        tmp = -1
    points[1][1] == tmp

    return points


def represent_partial_comp(comp):
    """ representation of partial equilibrium components, returns str"""
    ins_cp = None
    for point in comp:  # make sure all of them
        for i in range(3):
            point[i] = to_fraction(point[i])

    if len(comp) == 1:  # a point
        point = comp[0]
        ins_cp = IntersectionComponent(0, Interval(point[0], point[0]))
        ins_cp.functions.append(HypFracFunction(0, 1, HypFraction(0, point[1], 0, 1)))
        ins_cp.functions.append(HypFracFunction(0, 2, HypFraction(0, point[2], 0, 1)))

    elif len(comp) == 2:  # a segment
        changing_vars = []
        for i in range(3):
            if comp[0][i] != comp[1][i]:
                changing_vars.append(i)
        if len(changing_vars) == 0:
            point = comp[0]
            ins_cp = IntersectionComponent(0, Interval(point[0], point[0]))
            ins_cp.functions.append(HypFracFunction(0, 1, HypFraction(0, point[1], 0, 1)))
            ins_cp.functions.append(HypFracFunction(0, 2, HypFraction(0, point[2], 0, 1)))
        elif len(changing_vars) == 1:
            base = changing_vars[0]
            ins_cp = IntersectionComponent(base, Interval(comp[0][base], comp[1][base]))
            i1, i2 = (base+1) % 3, (base-1) % 3
            ins_cp.functions.append(HypFracFunction(base, i1, HypFraction(0, comp[0][i1], 0, 1)))
            ins_cp.functions.append(HypFracFunction(base, i2, HypFraction(0, comp[0][i2], 0, 1)))
        else:
            ("Error: unknown partial equi case")

    else:
        print("Error: unknown partial equi case")
    return str(ins_cp)


def get_extreme_points(c):
    # extreme points starting from (0,0) clockwise
    # None means on the surface
    ex_points = np.zeros((4, 3))

    points = [[0, 0], [0, 1], [1, 1], [1, 0]]

    for p in points:
        loc = is_on_BR(c, p[0], p[1])
        if loc == 0:
            p.append(np.nan)
        elif loc > 0:
            p.append(1)
        else:
            p.append(0)


def fraction_for_val(val, abcd):
    # f(val)= (a.val+b)/ (c.val+d)

    # denom
    if (vars[2]*val+vars[3]) != 0:
        return divide_in_fraction(vars[0]*val+vars[1], vars[2]*val+vars[3])
    else:
        return np.nan


def set_3D_settings(ax):
    # ax.set_xlabel('p')
    # ax.set_ylabel('q')
    # ax.set_zlabel('r')

    ax.set_xlim([0, 1.26])
    ax.set_ylim([0, 1.26])
    ax.set_zlim([1, 0])
    ax.view_init(16, -128)

    # Hide grid lines
    plt.axis('off')
    ax.grid(False)

    # ax.text(1.05, 0, 0, "p", None)
    # ax.text(0, 1.05, -0.05, "q", None)
    # ax.text(0, 0, 1.1, "r", None)
    # ax.text(0, -0.03, -0.07, "0", None)
    ax.text(1.05, 0, 0, "q", None, fontsize='x-large')
    ax.text(0, 1.05, -0.05, "r", None, fontsize='x-large')
    ax.text(0, 0, 1.1, "p", None, fontsize='x-large')
    ax.text(0, 0, -0.07, "0", None, fontsize='x-large')

    # full screen plot
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # draw borders of the cube
    color = 'darkgray'
    colorAx = 'dimgray'
    lnwidth = 1
    lnwidthAx = 1.5

    ax.plot3D([1, 1], [0, 1], [0, 0], color=color, linewidth=lnwidth)
    ax.plot3D([1, 0], [1, 1], [0, 0], color=color, linewidth=lnwidth)

    ax.plot3D([0, 1], [0, 0], [1, 1], color=color, linewidth=lnwidth)
    ax.plot3D([1, 1], [0, 1], [1, 1], color=color, linewidth=lnwidth)
    ax.plot3D([1, 0], [1, 1], [1, 1], color=color, linewidth=lnwidth)
    ax.plot3D([0, 0], [1, 0], [1, 1], color=color, linewidth=lnwidth)

    ax.plot3D([1, 1], [0, 0], [1, 0], color=color, linewidth=lnwidth)
    ax.plot3D([1, 1], [1, 1], [1, 0], color=color, linewidth=lnwidth)
    ax.plot3D([0, 0], [1, 1], [1, 0], color=color, linewidth=lnwidth)

    # white lines around ax lines
    lnwidthWhite = 5
    eps = 0.05

    ax.plot3D([0, 0], [1-eps, eps], [0, 0],
              color='white', linewidth=lnwidthWhite)
    ax.plot3D([0, 0], [0, 0], [1-eps, eps],
              color='white', linewidth=lnwidthWhite)
    ax.plot3D([eps, 1-eps], [0, 0], [0, 0],
              color='white', linewidth=lnwidthWhite)

    ax.plot3D([0, 0], [1, 0], [0, 0], color=colorAx, linewidth=lnwidthAx)
    ax.plot3D([0, 0], [0, 0], [1, 0], color=colorAx, linewidth=lnwidthAx)
    ax.plot3D([0, 1], [0, 0], [0, 0], color=colorAx, linewidth=lnwidthAx)


def get_all_cube_polygon():
    return np.array([[[0., 0., 0.],
                      [0., 1., 0.],
                      [1., 1., 0.],
                      [1., 0., 0.],
                      ],
                     [[1., 0., 1.],
                      [1., 1., 1.],
                      [0., 1., 1.],
                      [0., 0., 1.]
                      ],
                     [[0., 0., 0.],
                      [0., 0., 1.],
                      [1., 0., 1.],
                      [1., 0., 0.]
                      ],
                     [[1., 1., 0.],
                      [1., 1., 1.],
                      [0., 1., 1.],
                      [0., 1., 0.]
                      ],
                     [[0., 1., 0.],
                      [0., 1., 1.],
                      [0., 0., 1.],
                      [0., 0., 0.]
                      ],
                     [[1., 1., 0.],
                      [1., 1., 1.],
                      [1., 0., 1.],
                      [1., 0., 0.]]])


class BR_Shape(Enum):
    Hyperbola = 1
    DegHyperbola = 2
    Line = 3
    Constant = 4



class Intersection_Shape(Enum):
    AllCube = 0
    AllPlane = 1
    GenLine = 2
    VerLine = 3
    GenHype = 4
    VerHype = 5
    DegHype = 6
    Partial = 7
