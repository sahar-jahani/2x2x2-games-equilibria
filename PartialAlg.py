# Sahar Jahani

import numpy as np
import math
import modules as md

class PartialAlg:

    equation_variables = ['p', 'q', 'r']

    _K = np.zeros((3, 4))

    def __init__(self, payoff):
        """ 
        the payoff should not be the k-coefficients. if the payoffs are not normalised, normalised=false
        """
        self._normal_payoff_matrix = np.zeros((3, 4))
        self._mixed_eq_str = ""
        self._partial_eq_str = ""
        self._mixed_eq = []
        self._partial_eq = []
        self._equations_valid = np.ones(3, bool)

        
        self._normal_payoff_matrix = payoff

        # for i in range(3):
        #     if np.array_equal( self._normal_payoff_matrix[i],np.zeros(4)):
        #         raise ValueError("one player is completely indifferent")

    

    def get_payoff_matrix_from_input(self):
        pass

    def check_conditions(self):
        self.compute_coefficients()

        for i in range(3):
            # variables for this equation
            var1 = self.equation_variables[(i+1) % 3]
            var2 = self.equation_variables[(i-1) % 3]
            if self._K[i][0] == 0:
                if self._K[i][1] == 0:
                    self._mixed_eq_str += f"\n {i+1}: {var1} can NOT be written as func({var2})"
                    self._equations_valid[(i+1) % 3] = False
                else:  # denominator is valid so we can have a relation
                    self._mixed_eq_str += f"\n {i+1}: {var1} = ({-self._K[i][2]} {var2} + {-self._K[i][3]}) / {self._K[i][1]}"

                if self._K[i][2] == 0:
                    self._mixed_eq_str += f"\n {i+1}: {var2} can NOT be written as func({var1})"
                    self._equations_valid[(i-1) % 3] = False
                else:
                    self._mixed_eq_str += f"\n {i+1}: {var2} = ({-self._K[i][1]} {var1} + {-self._K[i][3]}) / {self._K[i][2]}"

            else:
                self._mixed_eq_str += f"\n {i+1}: {var1} = ({-self._K[i][2]} {var2} + {-self._K[i][3]}) / ({self._K[i][0]} {var2} + {self._K[i][1]})  "
                tmp = (-1*self._K[i][1]/self._K[i][0])
                if tmp >= 0 and tmp <= 1:
                    self._mixed_eq_str += f"EXCEPT if {var2} = {tmp} "

                self._mixed_eq_str += f"\n {i+1}: {var2} = ({-self._K[i][1]} {var1} + {-self._K[i][3]}) / ({self._K[i][0]} {var1} + {self._K[i][2]})  "
                tmp = (-1*self._K[i][2]/self._K[i][0])
                if tmp >= 0 and tmp <= 1:
                    self._mixed_eq_str += f"EXCEPT if {var1} = {tmp} "

                # self._mixed_eq += f"\n {i+1}: {var1} = ({-self._K[i][2]} {var2} + {-self._K[i][3]}) / ({self._K[i][0]} {var2} + {self._K[i][1]})  EXCEPT in {var2} = {(-1*self._K[i][1]/self._K[i][0])} "
                # self._mixed_eq += f"\n {i+1}: {var2} = ({-self._K[i][1]} {var1} + {-self._K[i][3]}) / ({self._K[i][0]} {var1} + {self._K[i][2]})  EXCEPT in {var1} = {(-1*self._K[i][2]/self._K[i][0])} "

    def compute_coefficients(self):
        for i in range(3):
            self._K[i][0] = self._normal_payoff_matrix[i][0]-self._normal_payoff_matrix[i][1] - \
                self._normal_payoff_matrix[i][2] + \
                self._normal_payoff_matrix[i][3]
            self._K[i][1] = self._normal_payoff_matrix[i][1] - \
                self._normal_payoff_matrix[i][0]
            self._K[i][2] = self._normal_payoff_matrix[i][2] - \
                self._normal_payoff_matrix[i][0]
            self._K[i][3] = self._normal_payoff_matrix[i][0]

    def compute_new_equations(self):
        
        # compute Quad equations if they are valid
        # print("equation valid:", self._equations_valid)
        
        new_coefs = [None, None, None]
        for i in range(3):
            new_coefs[i] = md.degHyperbola_to_lines(shape=None,c=self._K[i])

        for t0 in range(len(new_coefs[0])):
            for t1 in range(len(new_coefs[1])):
                for t2 in range(len(new_coefs[2])):
                    cur_coef = [new_coefs[0][t0],
                                new_coefs[1][t1], new_coefs[2][t2]]
                    self._mixed_eq_str += f"\n\n ---for K= {cur_coef}: \n"

                    equations_valid=np.ones(3)
                    for i in range(3):
                        if cur_coef[i][0] == 0:
                            if cur_coef[i][1] == 0:
                                equations_valid[(i+1) % 3] = False
                           
                            if self._K[i][2] == 0:
                                equations_valid[(i-1) % 3] = False

                    for i in range(3):
                        if equations_valid[i]:
                            x = cur_coef[i]
                            y = cur_coef[(i+1) % 3]
                            z = cur_coef[(i-1) % 3]

                            c_2 = (x[0]*z[1]*y[2]) - (x[1]*z[1]*y[0]) - \
                                (x[2]*z[0]*y[2]) + (x[3]*z[0]*y[0])
                            c_1 = ((x[0]*z[1]*y[3])+(x[0]*z[3]*y[2])) - ((x[1]*z[1]*y[1])+(x[1]*z[3]*y[0])
                                                                        ) - ((x[2]*z[0]*y[3])+(x[2]*z[2]*y[2]))+((x[3]*z[0]*y[1])+(x[3]*z[2]*y[0]))
                            c_0 = (x[0]*z[3]*y[3]) - (x[1]*z[3]*y[1]) - \
                                (x[2]*z[2]*y[3]) + (x[3]*z[2]*y[1])

                            self._mixed_eq_str += f"\n Quad({self.equation_variables[i]})= \t {c_2} {self.equation_variables[i]} ^2 + {c_1} {self.equation_variables[i]} + {c_0}"
                            solutions, res = self.compute_quadratic_roots(round(c_2, md.PERCISION_DECIMAL), round(c_1, md.PERCISION_DECIMAL), round(c_0, md.PERCISION_DECIMAL))
                            self._mixed_eq_str +=res
                            for sol in solutions:
                                val1 = None
                                val2 = None
                                if y[0]*sol+y[1] != 0:
                                    val2 = (-y[2]*sol-y[3]) / (y[0]*sol+y[1])
                                    if x[0]*val2+x[1] != 0:
                                        val1 = (-x[2]*val2-x[3])/(x[0]*val2+x[1])

                                if z[0]*sol+z[2] != 0:
                                    val1 = (-z[1]*sol-z[3])/(z[0]*sol+z[2])
                                    if x[0]*val1+x[2] != 0:
                                        val2 = (-x[1]*val1-x[3]) / (x[0]*val1+x[2])
                                if val1 is not None and val2 is not None:
                                    point = [0, 0, 0]
                                    point[i] = round(sol, md.PERCISION_DECIMAL)
                                    point[(i+1)%3] = round(val1, md.PERCISION_DECIMAL)
                                    point[(i-1)%3] = round(val2, md.PERCISION_DECIMAL)

                                    self._mixed_eq_str+=("\n"+ str(point))
                                    if md.is_in_cube(point):
                                        md.add_uniquly(self._mixed_eq, [point])

    def compute_quadratics_print(self)->str:
        # md.prt(f'k-coefficients matrix [MKLA]: \n{self._K}')
        # compute Quad equations if they are valid
        # print("equation_valid:", self._equations_valid)
        res=""
        for i in range(3):
            
            i1 = (i+1) % 3
            i2 = (i-1) % 3
            x = self._K[i]
            y = self._K[i1]
            z = self._K[i2]

            # c_2 = (x[0]*z[1]*y[2]) - (x[1]*z[1]*y[0]) - \
            #     (x[2]*z[0]*y[2]) + (x[3]*z[0]*y[0])
            # c_1 = ((x[0]*z[1]*y[3])+(x[0]*z[3]*y[2])) - ((x[1]*z[1]*y[1])+(x[1]*z[3]*y[0])
            #                                                 ) - ((x[2]*z[0]*y[3])+(x[2]*z[2]*y[2]))+((x[3]*z[0]*y[1])+(x[3]*z[2]*y[0]))
            # c_0 = (x[0]*z[3]*y[3]) - (x[1]*z[3]*y[1]) - \
            #     (x[2]*z[2]*y[3]) + (x[3]*z[2]*y[1])
            
            # coef of 2nd order
            c_2= round((np.linalg.det(np.array([[x[3],x[2],z[1]],[x[1],x[0],z[0]],[y[2],y[0],0]]))),md.PERCISION_DECIMAL)
            
            c_11= np.linalg.det(np.array([[x[3],x[2],z[3]],[x[1],x[0],z[2]],[y[2],y[0],0]]))
            c_12= np.linalg.det(np.array([[x[3],x[2],z[1]],[x[1],x[0],z[0]],[y[3],y[1],0]]))
            # coef of 1st order
            c_1= round(c_11+c_12, md.PERCISION_DECIMAL)
            # constant coef
            c_0= round((np.linalg.det(np.array([[x[3],x[2],z[3]],[x[1],x[0],z[2]],[y[3],y[1],0]]))),md.PERCISION_DECIMAL)
            

            res += f"\n Quad({self.equation_variables[i]})= \t {c_2} {self.equation_variables[i]} ^2 {'+' if c_1>=0 else '-'} {np.abs(c_1)} {self.equation_variables[i]} {'+' if c_0>=0 else '-'} {np.abs(c_0)}"
            _, roots = self.compute_quadratic_roots(c_2, c_1, c_0)
            res +=roots
            
        return res
            
            # for sol in solutions:
            #     val1 = None
            #     val2 = None
            #     if y[0]*sol+y[1] != 0:
            #         val2 = (-y[2]*sol-y[3]) / (y[0]*sol+y[1])
            #         if x[0]*val2+x[1] != 0:
            #             val1 = (-x[2]*val2-x[3])/(x[0]*val2+x[1])

            #     if z[0]*sol+z[2] != 0:
            #         val1 = (-z[1]*sol-z[3])/(z[0]*sol+z[2])
            #         if x[0]*val1+x[2] != 0:
            #             val2 = (-x[1]*val1-x[3]) / (x[0]*val1+x[2])
            #     if val1 is not None and val2 is not None:
            #         point = [0, 0, 0]
            #         point[i] = round(sol, md.PERCISION_DECIMAL)
            #         point[i1] = round(val1, md.PERCISION_DECIMAL)
            #         point[i2] = round(val2, md.PERCISION_DECIMAL)
            #         if md.is_in_cube(point):
            #             md.add_uniquly(self._mixed_eq, [point])
            
    def compute_ios_print(self)->str:
        # md.prt(f'k-coefficients matrix [MKLA]: \n{self._K}')
        # compute Quad equations if they are valid
        # print("equation_valid:", self._equations_valid)
        res=""
        for i in range(3):
            
            i1 = (i+1) % 3
            i2 = (i-1) % 3
            x = self._K[i]
            y = self._K[i1]
            z = self._K[i2]

            # IOS curve formulation:
            # c_yz yz + c_y y + c_z z + c_0 = 0
        
            c_yz= round(y[1]*z[0]-y[0]*z[2],md.PERCISION_DECIMAL)
            c_y= round(y[3]*z[0]-y[2]*z[2],md.PERCISION_DECIMAL)
            c_z= round(y[1]*z[1]-y[0]*z[3],md.PERCISION_DECIMAL)
            c_0= round(y[3]*z[1]-y[2]*z[3],md.PERCISION_DECIMAL)
            
            res += f"\n IOS({self.equation_variables[i1]},{self.equation_variables[i2]})= \t {c_yz} {self.equation_variables[i1]} {self.equation_variables[i2]} {'+' if c_y>=0 else '-'} {np.abs(c_y)} {self.equation_variables[i1]} {'+' if c_z>=0 else '-'} {np.abs(c_z)} {self.equation_variables[i2]} {'+' if c_0>=0 else '-'} {np.abs(c_0)}"
            
        return res

    def compute_quadratic_roots(self, a, b, c):
        ans = []
        result = ""
        if a == 0:
            if b == 0:
                if c == 0:
                    result += f"\n\t Infinite roots"
                else:
                    result += f"\n\t No root"
            else:
                result += f"\n\t Real root {(-c/b):.2f}"
                ans.append(round(-c/b,md.PERCISION_DECIMAL))
        else:
            dis_form = b * b - 4 * a * c
            sqrt_val = math.sqrt(abs(dis_form))

            if dis_form > 0:
                sol1 = round(((-b + sqrt_val) / (2 * a)),md.PERCISION_DECIMAL)
                sol2 = round(((-b - sqrt_val) / (2 * a)),md.PERCISION_DECIMAL)
                result += f"\n\t Real and different roots {sol1:.2f} and {sol2:.2f}"
                ans = [sol1, sol2]

            elif dis_form == 0:
                sol1 = round(((-b / (2 * a))),md.PERCISION_DECIMAL)
                result += f"\n\t Real and same roots {sol1:.2f}"
                ans = [sol1]
            else:
                sol=round(((-b / (2 * a))),md.PERCISION_DECIMAL)
                result += f"\n\t Complex roots {sol:.2f}  + i {sqrt_val:.2f} and {sol:.2f}  - i {sqrt_val:.2f}"

        self._mixed_eq_str += result
        return ans, result

    def compute_mixed_equilibria_algebraic(self):
        self.check_conditions()
        self.compute_new_equations()

        return md.section_format.format(title="Mixed Equilibria", answer=self._mixed_eq_str)

    def compute_partial_equilibria(self):
        for i in range(3):
            self.fix_player(i)

        return md.section_format.format(title="Partial Equilibria", answer=self._partial_eq_str)

    def fix_player(self, player):
        self.check_two_player_game(player, 0)
        self.check_two_player_game(player, 1)

    def check_two_player_game(self, fixed_player, fixed_action):
        NEs = []
        # finds equilibria when one player's action is fixed
        # locally define players for 2 player game
        P0 = (fixed_player+1) % 3
        P1 = (fixed_player+2) % 3
        # first action
        if fixed_action == 0:
            BR0 = md.compute_best_responce(
                0, self._normal_payoff_matrix[P0][0], self._normal_payoff_matrix[P0][1])
            BR1 = md.compute_best_responce(
                1, self._normal_payoff_matrix[P1][0], self._normal_payoff_matrix[P1][2])

        # second action
        elif fixed_action == 1:
            BR0 = md.compute_best_responce(
                0, self._normal_payoff_matrix[P0][2], self._normal_payoff_matrix[P0][3])
            BR1 = md.compute_best_responce(
                1, self._normal_payoff_matrix[P1][1], self._normal_payoff_matrix[P1][3])
        # print('BR0:',BR0)
        # print('BR1:',BR1)
        NEcandidates = md.compute_intersection_of_BRs(BR0, BR1)
        # print('intersection:',NEcandidates)
        # now it should be checked if they are NE even when fixed player changes
        NEs = self.check_partial_candidates(
            NEcandidates, fixed_player, fixed_action)

        self._partial_eq_str += f"\n fixed P{fixed_player+1} , action ={fixed_action}: ({NEs})"

        # print('NEs:', NEs)
        for NE in NEs:
            if NE is not None:
                item = []
                if len(NE) == 1:
                    point = [0, 0, 0]
                    point[fixed_player] = fixed_action
                    point[P0] = NE[0][0]
                    point[P1] = NE[0][1]
                    item = [point]
                elif len(NE) == 2:
                    # print(NE)
                    point0 = [0, 0, 0]
                    point1 = [0, 0, 0]

                    point0[fixed_player] = fixed_action
                    point1[fixed_player] = fixed_action

                    point0[P0] = NE[0][0]
                    point1[P0] = NE[1][0]

                    point0[P1] = NE[0][1]
                    point1[P1] = NE[1][1]

                    if(point0 == point1):
                        item = [point0]
                    else:
                        item = [point0, point1]

                md.add_uniquly(self._partial_eq, item)
                # print(self._partial_eq)

    def check_partial_candidates(self, candidates, player, action):
        NEs = []
        # payoff = self._normal_payoff_matrix[player]
        coef = self._K[player]
        # print('cands:',candidates)
        for item in candidates:
            if len(item) == 1 or (len(item) == 2 and item[0] == item[1]):
                # if player == 2 and action == 0:
                #     print('expected payoff curr:', self.compute_expected_payoff(
                #         player, action, item[0]))
                #     print('expected payoff other:', self.compute_expected_payoff(
                #         player, 1 - action, item[0]))
                if (self.compute_expected_payoff(player, action, item[0]) >= self.compute_expected_payoff(player, 1-action, item[0])):
                    NEs.append(md.validate_NE_point(item[0]))
            elif len(item) == 2:
                #Upoint_(current or other)
                U0_c = self.compute_expected_payoff(player, action, item[0])
                U0_o = self.compute_expected_payoff(player, 1-action, item[0])
                U1_c = self.compute_expected_payoff(player, action, item[1])
                U1_o = self.compute_expected_payoff(player, 1-action, item[1])
                if (U0_c >= U0_o):
                    if (U1_c >= U1_o):
                        NEs.append(md.validate_NE_segment(item[0], item[1]))
                    else:
                        # finds the segment that is the best response to the other player
                        # segment x=sth
                        if item[0][0] == item[1][0]:
                            mid = md.divide_in_fraction(
                                (-coef[3]-coef[1]*item[0][0]), (coef[2]+coef[0]*item[0][0]))
                            if mid > min(item[0][1], item[1][1]) and mid < max(item[0][1], item[1][1]):
                                NEs.append(md.validate_NE_segment(
                                    item[0], [item[0][0], mid]))
                                #NEs += [[item[0], [item[0][0], mid]]]
                        # segment y=sth
                        elif item[0][1] == item[1][1]:
                            mid = md.divide_in_fraction(
                                (-coef[3]-coef[2]*item[0][1]), (coef[1]+coef[0]*item[0][1]))
                            if mid > min(item[0][0], item[1][0]) and mid < max(item[0][0], item[1][0]):
                                NEs.append(md.validate_NE_segment(
                                    item[0], [mid, item[0][1]]))
                elif (U0_c <= U0_o) and (U1_c >= U1_o):
                    # segment x=sth
                    if item[0][0] == item[1][0]:
                        mid = md.divide_in_fraction(
                            (-coef[3]-coef[1]*item[0][0]), (coef[2]+coef[0]*item[0][0]))
                        if mid > min(item[0][1], item[1][1]) and mid < max(item[0][1], item[1][1]):
                            NEs.append(md.validate_NE_segment(
                                [item[0][0], mid], item[1]))
                    # segment y=sth
                    elif item[0][1] == item[1][1]:
                        tmp1 = (-coef[3]-coef[2]*item[0][1])
                        tmp2 = (coef[1]+coef[0]*item[0][1])
                        mid = md.divide_in_fraction(
                            (-coef[3]-coef[2]*item[0][1]), (coef[1]+coef[0]*item[0][1]))
                        if mid > min(item[0][0], item[1][0]) and mid < max(item[0][0], item[1][0]):
                            NEs.append(md.validate_NE_segment(
                                [mid, item[0][1]], item[1]))
        return NEs

    def compute_expected_payoff(self, main_player, action, probs):
        # print('prob: ',probs)
        # P0 = (main_player+1) % 3
        # P1 = (main_player+2) % 3

        if action == 0:
            return 0
        elif action == 1:
            # print("prob: ",probs)
            return ((self._normal_payoff_matrix[main_player][0])*(1-probs[0])*(1-probs[1])) + ((self._normal_payoff_matrix[main_player][1])*(probs[0])*(1-probs[1])) + (
                (self._normal_payoff_matrix[main_player][2])*(1-probs[0])*(probs[1])) + ((self._normal_payoff_matrix[main_player][3])*(probs[0])*(probs[1]))

    


