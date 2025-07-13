# Sahar Jahani
# This code computes the Nash equilibria of 2x2x2 games. Best-reponse correspondence and equilibria will be represented in a 3D plot  as well. 

import numpy as np
from PartialAlg import PartialAlg
from GamePlot import GamePlot
import fractions
import sys
import modules as md
import time
import matplotlib.pyplot as plt


md.initialize()
# command line arguments example: 
args_format= f"""
Please enter the payoff in one of the following formats

    normalized form payoff:                             -n [A,B,C,D] [a,b,c,d] [alpha,beta,gamma,delta]
    k-coefficients payoff:                              -k [M,K,L,A] [m,k,l,a] [mu,kappa,lambda,alpha]
    product payoffs (p-p1)(q-q2)-K1 etc.:               -r [p1,q2,K1] [r1,p2,K2] [q1,r2,K3]
    defined tests (0 to {len(md.TESTS)-1}):             -t testID/No
    from file (default game.stf):                       -f inputfile
    
options can be specified as follows

    verbose (default 1 print, 0 no print):              -v 0
    save as gif (default 0 not save, 1 save then name): -g 1 name
    orientation (default 1 horizontal, 0 vertical):     -o 0
    write output to file (default out.txt):             -w outputfile
    show plot (default 1 show, 0 not show):             -p 0
    save plot (default 0 not save, 1 save):             -s 1 name
    only plot best response of one player (1,2,3):      -b player
[enter options, -q to quit]
"""

def plot_actions(graph):
    if md.SAVE_PLOT:
        plt.savefig(md.SAVE_PLOT_NAME)
        print(f"plot {md.SAVE_PLOT_NAME} is saved")
        
    if md.SHOW_PLOT:
        print("\nclose the plot window to exit")
        plt.show()
        
    if md.GIF:
        print(f"\nsaving plot as gif ...")
        graph.rotate_plot(md.GIFNAME)
        print(f"gif {md.GIFNAME} is saved")
        
def solve_and_show_all(payoff):
    game = PartialAlg(payoff=payoff)
    game.compute_coefficients()
    md.prt(md.section_format.format(title='Normalized payoffs [ABCD]',answer=""))
    for i, pay in enumerate(game._normal_payoff_matrix.tolist()):
        tmp_list=[]
        for p in pay:
            tmp_list.append( md.to_fraction(p))
        md.prt(md.comp_format.format(ind=f"player {i+1}", comp= tmp_list))
        
    md.prt(md.section_format.format(title='Expected Payoff equations [MKLA]',answer=""))
    for i, pay in enumerate(game._K):
        i1=(i+1)%3
        i2=(i-1)%3
        equation= f"{md.to_fraction(pay[0])} {md._var_names[i1]}{md._var_names[i2]} + {md.to_fraction(pay[1])} {md._var_names[i1]} + {md.to_fraction(pay[2])} {md._var_names[i2]} + {md.to_fraction(pay[3])} = 0"
        md.prt(md.comp_format.format(ind=f"player {i+1}", comp= equation))
    
    
    game.compute_partial_equilibria()
    game.compute_mixed_equilibria_algebraic()
    
    graph = GamePlot(game._K, horizontal=md.ORIENTATION)

    if md.GIF and md.ORIENTATION:
        graph._fig_size=(16,10)
    elif md.GIF and not md.ORIENTATION:
        graph._fig_size=(10,16)
        
    graph.plot_all(game._partial_eq)
    
    md.prt(md.section_format.format(title="Pure and Partially Mixed equilibrium components",answer=len(game._partial_eq)))
    for i,cp in enumerate(game._partial_eq):
        repre_eq= md.represent_partial_comp(cp)
        md.prt(md.comp_format.format(ind=i+1, comp=repre_eq))

    md.prt(md.section_format.format(title="Completely Mixed equilibrium components",answer=len(graph._mixed_eq)))
    for i, cp in enumerate(graph._mixed_eq):
        md.prt(md.comp_format.format(ind=i+1, comp=cp))
        
    md.prt(md.section_format.format(title="Quadratic equations",answer=game.compute_quadratics_print()))    
    md.prt(md.section_format.format(title="IOS equations",answer=game.compute_ios_print()))    
        
    md.prt(md.section_format.format(title="Parameter relations",answer='\n'+game._mixed_eq_str))
    # md.prt(md.section_format.format(title="Completely Mixed equilibria computed algebraically",answer=len(game._mixed_eq)))
    # for i,cp in enumerate(game._mixed_eq):
    #     repre_eq= md.represent_partial_comp(cp)
    #     md.prt(md.comp_format.format(ind=i+1, comp=repre_eq))
        
    md.print_output()
    
    plot_actions(graph)


def plot_player( payoff, player, equ_componenets=None):
    game = PartialAlg(payoff=payoff)
    game.compute_coefficients()
    graph = GamePlot(game._K)

    graph.plot_player(player, equ_componenets)
    plot_actions(graph)
    



    
payoff= md.parse_arguments(sys.argv)

while payoff is None:
    print("Please enter the payoff in the following format \n"+args_format)
    args=input().split(' ')
    payoff= md.parse_arguments(args)
if md.ONLY_PLAYER is None:
    solve_and_show_all(payoff)
else:
    plot_player(payoff,player=md.ONLY_PLAYER)

    






