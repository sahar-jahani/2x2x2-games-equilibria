# Sahar Jahani

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from modules import HypFracFunction, HypFracSegment, IntersectionComponent
from modules import HypFraction
from modules import Interval
import modules as md

computation_accuracy = 10**6


class GamePlot:

    # _labels = ['p', 'q', 'r']
    _colors = [(1, 0, 0, 0.5), (0, 0, 1, 0.5), (0, 0.5, 0, 0.5)]
    _colors_names = ['red', 'blue', 'green']
    #[mixed eq, partial eq, intersection of 2 BRs]
    _intersection_colors=[(0,0,0,0.8), (0.3,0.3,0.3,0.8), (0.55,0,0.55,0.8)]
    # _color_patrial='gold'
    _fig_position = [2, 3, 4]
    _fig_size = (16, 7)
    _data_num = 50
    _hyperbola_accuracy = 20
    _intersection_accuracy = 30

    # player_BR_shape = [None, None, None]

    def __init__(self, K_coefficients, horizontal=True):
        self._mixed_eq = []
        self._fig = None
        self._3D_ax = None
        self._3D_all_axes = []
        self._player_shapes = [None, None, None]
        self._player_segments = [None, None, None]
        self._horizontal = horizontal

        self._coefficients = K_coefficients

        if horizontal:
            self._fig_size = (16, 7)
        else:
            self._fig_size = (10, 16)

    def plot_all(self, partial_eq):

        if self._horizontal:
            plotN = 230
        else:
            plotN = 320

        self._fig = plt.figure(figsize=self._fig_size)
        self._fig.subplots_adjust(hspace=0, wspace=0)
        ax = (self._fig).add_subplot(plotN+1, projection='3d')
        self._3D_ax = ax
        self._3D_all_axes.append(ax)
        md.set_3D_settings(ax)

        all_surfaces = []
        surfaces_player = []

        intersects = []

        for player in [0, 1, 2]:
            # tmp = self.plot_3D_player(ax, player, 322+player)
            intersect, mixed_comps = self.compute_intersection_2_BRs((player+1) % 3, (player-1) % 3)
            intersects.append(intersect)
            axSub = (self._fig).add_subplot(plotN+2+player, projection='3d')
            self._3D_all_axes.append(axSub)
            md.set_3D_settings(axSub)
            polygon,_ = self.compute_BR_polygon_player(player)
            self.draw_BR_polygon_player(polygon, self._colors[player], ax, axSub)
            axSub.add_collection3d(mplot3d.art3d.Poly3DCollection(
                intersect, color=self._intersection_colors[2], linewidth=5))

        axintsct = (self._fig).add_subplot(plotN+5, projection='3d')
        md.set_3D_settings(axintsct)
        self._3D_all_axes.append(axintsct)
        for i in range(len(intersects)):
            axintsct.add_collection3d(mplot3d.art3d.Poly3DCollection(
                intersects[i], color=self._colors[i], linewidth=5))

        axfin = (self._fig).add_subplot(plotN+6, projection='3d')
        md.set_3D_settings(axfin)
        self._3D_all_axes.append(axfin)
        mixd_poly, mixed_comps = self.compute_intersection_all_BRs()
        self._mixed_eq = mixed_comps
        # print("\n completely mixed equilibria: \n", comps,"\n", mixed_eq)
        self.plot_3D_components(mixd_poly, self._intersection_colors[0])
        
        partial_eq_qrp_coordinates=[]
        for cp in partial_eq:
            tmp=[c[:] for c in cp]
            md.change_representation_xyz_to_qrp(tmp)
            partial_eq_qrp_coordinates.append(tmp)
            
        self.plot_3D_components(partial_eq_qrp_coordinates, self._intersection_colors[1])
            


    def rotate_plot(self, gif_name='test_gif.gif', angle_step=3, fps=13):

        def rotate(angle):
            for axi in self._3D_all_axes:
                axi.view_init(azim=(-128+angle) % 360)

        ani = animation.FuncAnimation(self._fig, rotate, frames=np.arange(0, 360, angle_step), interval=50)
        ani.save(gif_name, writer=animation.PillowWriter(fps=fps))

    def plot_player(self, player, equ_components=None):

        self._fig = plt.figure(figsize=self._fig_size)
        self._fig.subplots_adjust(hspace=0, wspace=0)
        
        ax = (self._fig).add_subplot(111, projection='3d')
        self._3D_ax = ax
        self._3D_all_axes.append(ax)
        md.set_3D_settings(ax)

        all_surfaces = []
        surfaces_player = []
        
        tmp = self.plot_3D_player(ax, player)
        if tmp is not None:
            all_surfaces.append(tmp)
            surfaces_player.append(player)

        if equ_components is not None:
            self.plot_3D_components(equ_components, self._intersection_colors[1])
    # def plot_3D(self, players, equ_components=None, subplots=False):

    #     self._fig = plt.figure(figsize=self._fig_size)
    #     self._fig.subplots_adjust(hspace=0, wspace=0)
    #     if subplots:
    #         ax = (self._fig).add_subplot(221, projection='3d')
    #     else:
    #         ax = (self._fig).add_subplot(111, projection='3d')
    #     self._3D_ax = ax
    #     self._3D_all_axes.append(ax)
    #     md.set_3D_settings(ax)

    #     all_surfaces = []
    #     surfaces_player = []

    #     for player in players:
    #         pltIndex = None
    #         if subplots:
    #             pltIndex = 222+player
    #         tmp = self.plot_3D_player(ax, player, pltIndex)
    #         if tmp is not None:
    #             all_surfaces.append(tmp)
    #             surfaces_player.append(player)

    #     if equ_components is not None:
    #         self.plot_3D_components(equ_components, self._intersection_colors[1])

    #     # plt.show()

    def plot_3D_components(self, comps, color):


        for ax in self._3D_all_axes:
            for cp in comps:

                if len(cp) == 1:
                    ax.scatter3D(
                        cp[0][0], cp[0][1], cp[0][2], color=color, s=50)
                elif len(cp) == 2:
                    ax.plot3D([cp[0][0], cp[1][0]], [cp[0][1], cp[1][1]], [
                        cp[0][2], cp[1][2]], color=color, linewidth=5)
                else:
                    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(
                comps, color=color, linewidth=5))
                    break

    def plot_3D_player(self, ax, player, pltIndex=None):
        axSub = None
        if pltIndex is not None:
            axSub = (self._fig).add_subplot(pltIndex, projection='3d')
            self._3D_all_axes.append(axSub)
            md.set_3D_settings(axSub)
        polygon,_ = self.compute_BR_polygon_player(player)
        self.draw_BR_polygon_player(polygon, self._colors[player], ax, axSub)

    def draw_BR_polygon_player(self, polygon, color, ax, axSub=None):
        ax.add_collection3d(mplot3d.art3d.Poly3DCollection(
            polygon, color=color))
        if axSub is not None:
            axSub.add_collection3d(mplot3d.art3d.Poly3DCollection(
                polygon, color=color))

    def compute_BR_polygon_player(self, player):
        c = self._coefficients[player]
        shape, all_points, surf_points, surf_points_index = md.compute_shape_and_points(
            c)

        # print(all_points)
        surf_n = len(surf_points)
        all_n = len(all_points)
        all_poly = []
        # identify the polygons
        if np.array_equal(np.array(c), np.zeros(4)):
            all_poly = md.get_all_cube_polygon()
        elif surf_n == 0:
            shape = md.BR_Shape.Constant
            all_points = md.modify_points_ax(all_points, player)
            all_poly.append(all_points)
        elif surf_n == 1:

            all_poly.append(md.prepare_mapped_surface(all_points, player))
            all_poly.append(md.prepre_BR_main([surf_points[0].copy()], player))

        else:
            if (surf_n == 2):
                if shape == md.BR_Shape.Line or shape == md.BR_Shape.DegHyperbola:
                    i = surf_points_index[0]
                    j = surf_points_index[1]

                    if (j-i) < (all_n-1):
                        part1 = []
                        for k in range(i+1):
                            part1.append(all_points[k].copy())

                        for k in range(j, all_n):
                            part1.append(all_points[k].copy())
                        all_poly.append(md.prepare_mapped_surface(part1, player))

                    if (j-i) > 1:
                        part2 = []
                        for k in range(i, j+1):
                            part2.append(all_points[k].copy())
                        all_poly.append(md.prepare_mapped_surface(part2, player))

                    all_poly.append(md.prepre_BR_main(surf_points, player))

                elif shape == md.BR_Shape.Hyperbola:
                    if not (md.same_arc_of_hyperbola(c, surf_points[0], surf_points[1])):
                        all_poly.append(md.prepre_BR_main(
                            [surf_points[0]], player))
                        all_poly.append(md.prepre_BR_main(
                            [surf_points[1]], player))
                        all_poly.append(
                            md.prepare_mapped_surface(all_points, player))
                    else:
                        direction = 1
                        if surf_points[0][0] < surf_points[1][0]:
                            hyp = md.get_equal_distance_hyperbola_points(c,
                                                                         surf_points[0][0], surf_points[1][0], self._hyperbola_accuracy)
                        else:
                            hyp = md.get_equal_distance_hyperbola_points(c,
                                                                         surf_points[1][0], surf_points[0][0], self._hyperbola_accuracy)
                            direction = -1

                        i = surf_points_index[0]
                        j = surf_points_index[1]

                        part1 = []
                        for k in range(i):
                            part1.append((all_points[k].copy()))

                        if direction == 1:
                            part1 = part1+(hyp.copy())
                        else:
                            hyp_re = hyp.copy()
                            hyp_re.reverse()
                            part1 = part1+hyp_re
                        for k in range(j+1, all_n):
                            part1.append((all_points[k].copy()))
                        all_poly.append(md.prepare_mapped_surface(part1, player))

                        part2 = []
                        for k in range(i+1, j):
                            part2.append((all_points[k].copy()))

                        if direction != 1:
                            part2 = part2+(hyp.copy())
                        else:
                            hyp_re = hyp.copy()
                            hyp_re.reverse()
                            part2 = part2+hyp_re

                        all_poly.append(md.prepare_mapped_surface(part2, player))

                        all_poly = all_poly + \
                            (md.create_hyperbola_small_surfaces(hyp, player))

            elif (surf_n == 3):
                if shape == md.BR_Shape.DegHyperbola:
                    center = [
                        md.divide_in_fraction(-c[2], c[0]), md.divide_in_fraction(-c[1], c[0])]
                    part1 = []
                    part2 = []
                    for k in range(3):
                        if surf_points[k][0] == center[0]:
                            part1.append((surf_points[k]).copy())
                        if surf_points[k][1] == center[1]:
                            part2.append((surf_points[k]).copy())

                    all_poly.append(md.prepare_mapped_surface(all_points, player))
                    all_poly.append(md.prepre_BR_main(part1, player))
                    all_poly.append(md.prepre_BR_main(part2, player))

                elif shape == md.BR_Shape.Hyperbola:

                    is_found = False
                    arc_ind = [-1, -1]
                    pnt_ind = -1
                    for k in range(3):
                        k_ne = (k+1) % 3
                        if md.same_arc_of_hyperbola(c, surf_points[k], surf_points[k_ne]):
                            is_found = True
                            pnt_ind = (k-1) % 3
                            if (k_ne > k):
                                arc_ind[0] = k
                                arc_ind[1] = k_ne
                            else:
                                arc_ind[0] = k_ne
                                arc_ind[1] = k
                            break
                    if is_found:
                        part1 = []
                        part2 = []
                        part3 = []

                        direction = 1
                        if surf_points[arc_ind[0]][0] < surf_points[arc_ind[1]][0]:
                            hyp = md.get_equal_distance_hyperbola_points(c,
                                                                         surf_points[arc_ind[0]][0], surf_points[arc_ind[1]][0], self._hyperbola_accuracy)
                        else:
                            hyp = md.get_equal_distance_hyperbola_points(c,
                                                                         surf_points[arc_ind[1]][0], surf_points[arc_ind[0]][0], self._hyperbola_accuracy)
                            direction = -1

                        if direction == -1:
                            hyp.reverse()

                        hyp_start_ind = surf_points_index[arc_ind[0]]
                        hyp_end_ind = surf_points_index[arc_ind[1]]
                        for k in range(hyp_start_ind):
                            part1.append((all_points[k]).copy())
                        part1 = part1+(hyp.copy())
                        for k in range(hyp_end_ind+1, all_n):
                            part1.append((all_points[k]).copy())
                        all_poly.append(md.prepare_mapped_surface(part1, player))

                        for k in range(hyp_start_ind, hyp_end_ind+1):
                            part2.append((all_points[k]).copy())
                        hyp2 = hyp.copy()
                        hyp2.reverse()
                        part2 = part2+hyp2
                        all_poly.append(md.prepare_mapped_surface(part2, player))
                        all_poly = all_poly + \
                            (md.create_hyperbola_small_surfaces(hyp, player))

                        all_poly.append(md.prepre_BR_main(
                            [surf_points[pnt_ind].copy()], player))

                    else:
                        print(
                            "compute_BR_polygon: Error in finding hyperbola arc points", surf_points)

            elif (surf_n == 4):
                center = [
                    md.divide_in_fraction(-c[2], c[0]), md.divide_in_fraction(-c[1], c[0])]
                if shape == md.BR_Shape.DegHyperbola:
                    hor_pnts_ind = []
                    ver_pnts_ind = []
                    for k in range(len(surf_points)):
                        if surf_points[k][0] == center[0]:
                            ver_pnts_ind.append(k)
                        if surf_points[k][1] == center[1]:
                            hor_pnts_ind.append(k)

                    if md.is_in_interior(center[0], center[1]):
                        center_c = center+[np.inf]
                        for k in range(3):
                            part = []
                            for l in range(surf_points_index[k], surf_points_index[k+1]+1):
                                part.append(all_points[l])
                            part.append(center_c)
                            all_poly.append(
                                md.prepare_mapped_surface(part, player))
                        part_last = [all_points[surf_points_index[3]],
                                     all_points[0], all_points[surf_points_index[0]], center_c]
                        all_poly.append(
                            md.prepare_mapped_surface(part_last, player))

                        all_poly.append(md.prepre_BR_main(
                            [surf_points[0], surf_points[2]], player))
                        all_poly.append(md.prepre_BR_main(
                            [surf_points[1], surf_points[3]], player))

                    else:
                        main_surf2 = []
                        if (center[1].denominator) != 1 and len(hor_pnts_ind) == 2:
                            i = surf_points_index[hor_pnts_ind[0]]
                            j = surf_points_index[hor_pnts_ind[1]]
                            for k in ver_pnts_ind:
                                main_surf2.append(
                                    all_points[surf_points_index[k]])
                        elif (center[0].denominator) != 1 and len(ver_pnts_ind) == 2:
                            i = surf_points_index[ver_pnts_ind[0]]
                            j = surf_points_index[ver_pnts_ind[1]]
                            for k in hor_pnts_ind:
                                if md.is_pure_strategy(all_points[surf_points_index[k]], 2):
                                    main_surf2.append(
                                        all_points[surf_points_index[k]])
                        else:
                            print("Error in surface_n==4 for player", player)
                            return None

                        main_surf1 = [all_points[i], all_points[j]]
                        all_poly.append(md.prepre_BR_main(main_surf1, player))

                        all_poly.append(md.prepre_BR_main(main_surf2, player))

                        part1 = []
                        for k in range(i):
                            part1.append(all_points[k])
                        part1 = part1+main_surf1
                        for k in range(j+1, all_n):
                            part1.append(all_points[k])
                        all_poly.append(md.prepare_mapped_surface(part1, player))

                        part2 = []
                        for k in range(i+1, j):
                            part2.append(all_points[k])
                        main_surf1.reverse()
                        part2 = part2+main_surf1
                        all_poly.append(md.prepare_mapped_surface(part2, player))

                elif shape == md.BR_Shape.Hyperbola:
                    x_sort_ind = np.arange(0, 4, 1)
                    for k in range(4):
                        for l in range(k+1, 4):
                            if surf_points[x_sort_ind[k]][0] > surf_points[x_sort_ind[l]][0]:
                                tmp = x_sort_ind[k]
                                x_sort_ind[k] = x_sort_ind[l]
                                x_sort_ind[l] = tmp
                    x_0 = x_sort_ind[0]
                    x_1 = x_sort_ind[1]
                    x_2 = x_sort_ind[2]
                    x_3 = x_sort_ind[3]

                    hyp1 = md.get_equal_distance_hyperbola_points(c,
                                                                  surf_points[x_0][0], surf_points[x_1][0], self._hyperbola_accuracy)
                    hyp2 = md.get_equal_distance_hyperbola_points(c,
                                                                  surf_points[x_2][0], surf_points[x_3][0], self._hyperbola_accuracy)

                    all_poly = all_poly + \
                        (md.create_hyperbola_small_surfaces(hyp1, player))
                    all_poly = all_poly + \
                        (md.create_hyperbola_small_surfaces(hyp2, player))

                    form = 1
                    if x_1 == 3:
                        form = -1

                    if form == 1:
                        part1 = []
                        for k in range(surf_points_index[x_0]):
                            part1.append(all_points[k])
                        part1 = part1+hyp1
                        for k in range(surf_points_index[x_1]+1, surf_points_index[x_3]):
                            part1.append(all_points[k])
                        hyp2.reverse()
                        part1 = part1+hyp2
                        for k in range(surf_points_index[x_2]+1, all_n):
                            part1.append(all_points[k])
                        all_poly.append(md.prepare_mapped_surface(part1, player))

                        part2 = []
                        for k in range(surf_points_index[x_0], surf_points_index[x_1]):
                            part2.append(all_points[k])
                        hyp1.reverse()
                        part2 = part2+hyp1
                        all_poly.append(md.prepare_mapped_surface(part2, player))

                        part3 = []
                        for k in range(surf_points_index[x_3], surf_points_index[x_2]):
                            part3.append(all_points[k])
                        hyp2.reverse()
                        part3 = part3+hyp2
                        all_poly.append(md.prepare_mapped_surface(part3, player))
                    else:
                        part1 = [all_points[surf_points_index[x_1]],
                                 all_points[0], all_points[surf_points_index[x_0]]]
                        part1 = part1+hyp1
                        all_poly.append(md.prepare_mapped_surface(part1, player))

                        part2 = []
                        for k in range(surf_points_index[x_2], surf_points_index[x_3]):
                            part2.append(all_points[k])
                        hyp2.reverse()
                        part2 = part2+hyp2
                        all_poly.append(md.prepare_mapped_surface(part2, player))

                        part3 = []
                        hyp1.reverse()
                        hyp2.reverse()
                        part3 = part3+hyp1
                        for k in range(surf_points_index[x_0], surf_points_index[x_2]):
                            part3.append(all_points[k])
                        part3 = part3+hyp2
                        for k in range(surf_points_index[x_3], surf_points_index[x_1]):
                            part3.append(all_points[k])
                        all_poly.append(md.prepare_mapped_surface(part3, player))

        self._player_shapes[player] = shape
        return all_poly, [f"Best Response of player {player}: {shape.name}"]

    def compute_intersection_2_BRs(self, player1, player2):
        comps = []
        for i in [player1, player2]:
            if self._player_shapes[i] is None:
                self.compute_BR_polygon_player(i)

        p1 = player1
        p2 = player2
        if ((p1+1) % 3) != ((p2-1) % 3):
            p1 = player2
            p2 = player1

        p_common = 3-(p1+p2)

        c1 = self._coefficients[p1]
        c2 = self._coefficients[p2]

        # check extreme cases
        if np.array_equal(np.array(c1), np.zeros(4)) and np.array_equal(np.array(c2), np.zeros(4)):
            # should return all the cube

            return md.get_all_cube_polygon(), None
        elif np.array_equal(np.array(c1), np.zeros(4)):
            # last_shape = player_shape_to_intersection_shape(
            #     self._player_shapes[p2])
            return self.compute_BR_polygon_player(p2)
        elif np.array_equal(np.array(c2), np.zeros(4)):
            # last_shape = player_shape_to_intersection_shape(
            #     self._player_shapes[p1])
            return self.compute_BR_polygon_player(p1)

        else:
            parts1 = md.get_BR_function_segments(c1)
            parts2 = md.get_BR_function_segments(c2)

            for prt1 in parts1:
                for prt2 in parts2:
                    if (prt2.p1Fp2 is None) and (prt1.p2Fp1 is None):
                        if (prt2.p2Fp1).is_equal(prt1.p1Fp2):
                            tmp = IntersectionComponent(
                                p_common, prt1.p1Intervals[0])
                            tmp.unres_vars = [p1, p2]
                            tmp.functions.append(
                                HypFracFunction(p1, p_common, prt1.p1Fp2))
                            comps.append(tmp)
                    elif (prt2.p1Fp2 is None):

                        for interval in prt1.p1Intervals:
                            intvl = interval.intersection(
                                prt2.p2Intervals[0])
                            if intvl is not None:
                                tmp = IntersectionComponent(p_common, intvl)
                                tmp.unres_vars = [p1]
                                tmp.functions.append(
                                    HypFracFunction(p_common, p2, prt1.p2Fp1))
                                comps.append(tmp)
                    elif (prt1.p2Fp1 is None):

                        for interval in prt2.p2Intervals:
                            intvl = interval.intersection(
                                prt1.p1Intervals[0])
                            if intvl is not None:
                                tmp = IntersectionComponent(p_common, intvl)
                                tmp.unres_vars = [p2]
                                tmp.functions.append(
                                    HypFracFunction(p_common, p1, prt2.p1Fp2))
                                comps.append(tmp)
                    else:
                        for in1 in prt1.p1Intervals:
                            for in2 in prt2.p2Intervals:
                                intvl = in1.intersection(in2)
                                if intvl is not None:
                                    tmp = IntersectionComponent(
                                        p_common, intvl)
                                    tmp.functions.append(
                                        HypFracFunction(p_common, p2, prt1.p2Fp1))
                                    tmp.functions.append(
                                        HypFracFunction(p_common, p1, prt2.p1Fp2))
                                    if (prt1.p1Fp2 is None) and (prt2.p2Fp1 is None):
                                        tmp.unres_vars.append(p_common)
                                    comps.append(tmp)

            poly = self.intersection_comps_to_polygon(comps)
            for cp in poly:
                md.change_representation_xyz_to_qrp(cp)
            return poly, comps

    def compute_intersection_all_BRs(self):
        comps = []
        for i in range(3):
            if self._player_shapes[i] is None:
                self.compute_BR_polygon_player(i)

        for i in range(3):
            if np.array_equal(np.array(self._coefficients[i]), np.zeros(4)):
                return self.compute_intersection_2_BRs((i+1) % 3, (i-1) % 3)

        try:
            _, comps = self.compute_intersection_2_BRs(0, 1)
            final_comps = self.intersect_2BRs_with_3rd(comps, 2)
            final_poly = self.intersection_comps_to_polygon(final_comps)
            for cp in final_poly:
                md.change_representation_xyz_to_qrp(cp)
            return final_poly, final_comps
        except:
            try:
                _, comps = self.compute_intersection_2_BRs(0, 2)
                final_comps = self.intersect_2BRs_with_3rd(comps, 1)
                final_poly = self.intersection_comps_to_polygon(final_comps)
                for cp in final_poly:
                    md.change_representation_xyz_to_qrp(cp)
                return final_poly, final_comps
            except:
                try:
                    _, comps = self.compute_intersection_2_BRs(1, 2)
                    final_comps = self.intersect_2BRs_with_3rd(comps, 0)
                    final_poly = self.intersection_comps_to_polygon(final_comps)
                    for cp in final_poly:
                        md.change_representation_xyz_to_qrp(cp)
                    return final_poly, final_comps
                except:
                    print("Error: None of intersection combinations work!")

    def intersect_2BRs_with_3rd(self, comps, player):
        res = []
        c = self._coefficients[player]
        p1 = (player+1) % 3
        p2 = (player-1) % 3
        segments = md.get_BR_function_segments(c)
        for segment in segments:
            for comp in comps:
                if player in comp.unres_vars:
                    # the other values have to be constant
                    point = np.zeros(3)
                    for f in comp.functions:
                        point[f.output_ind] = f.fraction.get_val(point[f.input_ind])
                    if md.is_on_BR(c, point[p1], point[p2]) == 0:
                        res.append(comp)
                elif len(comp.unres_vars) == 2:
                    # IOS is a plane, player value is constant so it will be player's BR with fixed player value

                    if segment.p1Fp2 is None:
                        for intvl in segment.p1Intervals:
                            intsct = IntersectionComponent(basic_var=p1, basic_var_interval=intvl)
                            intsct.functions.append(HypFracFunction(p1, p2, segment.p2Fp1))
                            intsct.functions.append((comp.functions[0]))
                            res.append(intsct)
                    else:
                        for intvl in segment.p2Intervals:
                            intsct = IntersectionComponent(basic_var=p2, basic_var_interval=intvl)
                            intsct.functions.append(HypFracFunction(p2, p1, segment.p1Fp2))
                            intsct.functions.append((comp.functions[0]))
                            res.append(intsct)
                elif len(comp.unres_vars) == 1:

                    if len(comp.functions) == 1 and (comp.functions[0]).input_ind == player and comp.basic_var_interval.len() == 0:
                        f = comp.functions[0]
                        basic_val = comp.basic_var_interval.s
                        other_val = f.fraction.get_val(basic_val)

                        # p2=const
                        if comp.unres_vars[0] == p1:
                            if (segment.p1Fp2 is None) and (other_val == segment.p2Fp1.get_val(0)):
                                res.append(comp)
                            elif (segment.p1Fp2 is not None) and md.is_in_interior(segment.p1Fp2.get_val(other_val)):
                                tmp = segment.p1Fp2.get_val(other_val)
                                intsct = IntersectionComponent(basic_var=p1, basic_var_interval=Interval(tmp, tmp))
                                intsct.functions.append(HypFracFunction(p1, p2, HypFraction(0, other_val, 0, 1)))
                                intsct.functions.append(HypFracFunction(p1, player, HypFraction(0, basic_val, 0, 1)))
                                res.append(intsct)
                        # p1=const
                        elif comp.unres_vars[0] == p2:
                            if (segment.p2Fp1 is None) and (other_val == segment.p1Fp2.get_val(0)):
                                res.append(comp)
                            elif (segment.p2Fp1 is not None) and md.is_in_interior(segment.p2Fp1.get_val(other_val)):
                                tmp = segment.p2Fp1.get_val(other_val)
                                intsct = IntersectionComponent(basic_var=p2, basic_var_interval=Interval(tmp, tmp))
                                intsct.functions.append(HypFracFunction(p2, p1, HypFraction(0, other_val, 0, 1)))
                                intsct.functions.append(HypFracFunction(p2, player, HypFraction(0, basic_val, 0, 1)))
                                res.append(intsct)

                        else:
                            raise Exception("undealt case with unres_vars!")

                    else:
                        raise Exception("undealt case!")

                else:
                    # no unres_var so just curves in comp

                    p1FuncBase = comp.functions[0]
                    p2FuncBase = comp.functions[1]

                    if comp.functions[0].output_ind == p2:
                        p2FuncBase = comp.functions[0]
                        p1FuncBase = comp.functions[1]

                    if segment.p1Fp2 is None:
                        # p2=const so p2=f(p1) in comp is needed
                        ftmp, p1intvl = p1FuncBase.fraction.switch_variable_and_interval(comp.basic_var_interval)
                        if ftmp is not None:
                            cmp_p2Fp1 = p2FuncBase.fraction.composition(ftmp)
                            intsc_p1 = segment.p2Fp1.intersection(cmp_p2Fp1)
                            if intsc_p1 is not None:
                                for p1_val in intsc_p1:
                                    if p1intvl.contains(p1_val):
                                        intsct = IntersectionComponent(p1, Interval(p1_val, p1_val))
                                        intsct.functions.append(HypFracFunction(p1, p2, segment.p2Fp1))
                                        intsct.functions.append(HypFracFunction(p1, player, ftmp))
                                        res.append(intsct)
                            else:
                                # means the quadratic equation has infinite solutions
                                res.append(comp)
                        else:
                            # segment p2=const1 comp: p1=const2, p2=f(player) -> if const1 in range of f: (const2,const1, inv(f(const1)))
                            ftmp, p2intvl = p2FuncBase.fraction.switch_variable_and_interval(comp.basic_var_interval)
                            if p2intvl.contains(segment.p2Intervals[0].s):
                                p2_val = segment.p2Intervals[0].s
                                intsct = IntersectionComponent(p2, Interval(p2_val, p2_val))
                                intsct.functions.append(HypFracFunction(p2, p1, p1FuncBase.fraction))
                                intsct.functions.append(HypFracFunction(p2, player, ftmp))
                                res.append(intsct)

                    else:
                        # segment p1fp2 not none, p1=H(p2), H can be const or hyp but no problem
                        baseFp2, p2intvl = p2FuncBase.fraction.switch_variable_and_interval(comp.basic_var_interval)
                        if baseFp2 is not None:
                            cmp_p1Fp2 = p1FuncBase.fraction.composition(baseFp2)
                            intsc_p2 = segment.p1Fp2.intersection(cmp_p1Fp2)
                            if intsc_p2 is not None:
                                for p2_val in intsc_p2:
                                    if p2intvl.contains(p2_val):
                                        intsct = IntersectionComponent(p2, Interval(p2_val, p2_val))
                                        intsct.functions.append(HypFracFunction(p2, p1, segment.p1Fp2))
                                        intsct.functions.append(HypFracFunction(p2, player, baseFp2))
                                        res.append(intsct)
                            else:
                                res.append(comp)
                        else:
                            # segment p1fp2 not none comp: p2=const, p1=f(player) , comp cant be a line because len(unres)=0 =-> (p1Fp2(const),const, f-inv(p1Fp2(const)))
                            for segp2intvl in segment.p2Intervals:
                                baseFp1, p1intvl = p1FuncBase.fraction.switch_variable_and_interval(
                                    comp.basic_var_interval)
                                _, segp1intvl = segment.p1Fp2.switch_variable_and_interval(segp2intvl)
                                baseFp2 = baseFp1.composition(segment.p1Fp2)
                                p2_val = p2intvl.s
                                # p2intvl should be the only value p2 can get, const
                                if segp2intvl.contains(p2_val) and p1intvl.does_intersect(segp2intvl) and comp.basic_var_interval.contains(baseFp2.get_val(p2_val)):
                                    p2_val = p2intvl.s
                                    intsct = IntersectionComponent(p2, Interval(p2_val, p2_val))
                                    intsct.functions.append(HypFracFunction(p2, p1, segment.p1Fp2))

                                    intsct.functions.append(HypFracFunction(p2, player, baseFp2))
                                    res.append(intsct)
                    # else:
                    #     #segment is a curve, comp is also a curve
                    #     baseFp1, p1intvl = p1Fbase.fraction.switch_variable_and_interval(comp.basic_var_interval)
                    #     cmp_p2Fp1 = p2Fbase.fraction.composition(baseFp1)
                    #     intsc_p2 = segment.p2Fp1.intersection(cmp_p2Fp1)
                    #     if intsc_p2 is not None:
                    #         find the points###############
                    #     else:
                    #         res.append(comp)

        return res

    def compute_hyperbola_intersection_arc_points(self, surf_points, coef, c1, c2, ch_ax_0, ch_ax_1):

        hyp1 = []
        hyp2 = []
        twoArcs = False
        surf_n = len(surf_points)
        for k in range(surf_n):
            for l in range(k+1, surf_n):
                if surf_points[k][ch_ax_0] > surf_points[l][ch_ax_0]:
                    tmp = surf_points[k]
                    surf_points[k] = surf_points[l]
                    surf_points[l] = tmp
        center_x = md.divide_in_fraction(-coef[2], coef[0])
        if surf_n == 4:
            points_x1 = np.linspace(
                surf_points[0][ch_ax_0], surf_points[1][ch_ax_0], self._hyperbola_accuracy)
            points_x2 = np.linspace(
                surf_points[2][ch_ax_0], surf_points[3][ch_ax_0], self._hyperbola_accuracy)
            twoArcs = True
        elif surf_n == 3:

            if surf_points[1][ch_ax_0] < center_x:
                points_x1 = np.linspace(
                    surf_points[0][ch_ax_0], surf_points[1][ch_ax_0], self._hyperbola_accuracy)
            else:
                points_x1 = np.linspace(
                    surf_points[1][ch_ax_0], surf_points[2][ch_ax_0], self._hyperbola_accuracy)
        elif surf_n == 2:
            if (surf_points[1][ch_ax_0] - center_x) * (surf_points[0][ch_ax_0] - center_x) > 0:
                points_x1 = np.linspace(
                    surf_points[0][ch_ax_0], surf_points[1][ch_ax_0], self._hyperbola_accuracy)
        else:
            return hyp1, hyp2

        for x in points_x1:
            val_ch1 = md.get_point_on_hyperbola(coef, x)
            other_ind = 3-(ch_ax_0+ch_ax_1)
            if ch_ax_0 == 0:
                val_other = (md.fraction_for_val(val_ch1, [-c1[2], -c1[3], c1[0], c1[1]])) if (not math.isnan(
                    md.fraction_for_val(x, [-c1[2], -c1[3], c1[0], c1[1]])))else (md.fraction_for_val(x, [-c2[1], -c2[3], c2[0], c2[2]]))
            else:
                val_other = surf_points[0][other_ind]
            point = [0, 0, 0]
            point[ch_ax_0] = x
            point[ch_ax_1] = val_ch1
            point[other_ind] = val_other

            hyp1.append(point.copy())
        if twoArcs:
            for x in points_x2:
                val_ch1 = md.get_point_on_hyperbola(coef, x)
                other_ind = 3-(ch_ax_0+ch_ax_1)
                if ch_ax_0 == 0:
                    val_other = (md.fraction_for_val(val_ch1, [-c1[2], -c1[3], c1[0], c1[1]])) if (not math.isnan(
                        md.fraction_for_val(x, [-c1[2], -c1[3], c1[0], c1[1]])))else (md.fraction_for_val(x, [-c2[1], -c2[3], c2[0], c2[2]]))
                else:
                    val_other = surf_points[0][other_ind]
                point = [0, 0, 0]
                point[ch_ax_0] = x
                point[ch_ax_1] = val_ch1
                point[other_ind] = val_other

                hyp2.append(point.copy())
        return hyp1, hyp2

    def intersection_comps_to_polygon(self, comps):
        poly = []
        for comp in comps:
            if len(comp.unres_vars) == 2:
                point = np.zeros(3)
                point[comp.basic_var] = comp.basic_var_interval.s
                poly.append(md.define_BR_surface(md.define_BR_surface(
                    [point], comp.unres_vars[0]), comp.unres_vars[1]))
            elif len(comp.unres_vars) < 2:
                points = []
                if comp.basic_var_interval.len() > 0:
                    basic_var_divided = np.linspace(
                        comp.basic_var_interval.s, comp.basic_var_interval.e, self._intersection_accuracy)
                else:
                    basic_var_divided = [comp.basic_var_interval.s]
                for i in range(len(basic_var_divided)):
                    part = []
                    if i > 0:
                        vals = [basic_var_divided[i-1], basic_var_divided[i]]
                    else:
                        vals = [basic_var_divided[i]]
                    for val in vals:
                        point = np.zeros(3)
                        point[comp.basic_var] = val
                        for func in comp.functions:
                            if func.input_ind == comp.basic_var:
                                point[func.output_ind] = func.fraction.get_val(
                                    val)
                        ##########################################################################################################
                        if md.is_in_cube(point):
                            part.append(point)

                    if len(comp.unres_vars) > 0 and len(part) > 0:
                        part = md.define_BR_surface(part, comp.unres_vars[0])
                    # if len(part)==1:
                    #     part=[part[0],part[0]]
                    if len(part) > 0:
                        poly.append(part)
            else:
                poly.append(md.get_all_cube_polygon())

        return poly

    # def compute_quadratic_solutions(self):
    #     for i in range(3):
    #         if self._equations_valid[i]:
    #             i1=(i+1) % 3
    #             i2=(i-1) % 3
    #             x = self._K[i]
    #             y = self._K[i1]
    #             z = self._K[i2]

    #             c_2 = (x[0]*z[1]*y[2]) - (x[1]*z[1]*y[0]) - \
    #                 (x[2]*z[0]*y[2]) + (x[3]*z[0]*y[0])
    #             c_1 = ((x[0]*z[1]*y[3])+(x[0]*z[3]*y[2])) - ((x[1]*z[1]*y[1])+(x[1]*z[3]*y[0])
    #                                                          ) - ((x[2]*z[0]*y[3])+(x[2]*z[2]*y[2]))+((x[3]*z[0]*y[1])+(x[3]*z[2]*y[0]))
    #             c_0 = (x[0]*z[3]*y[3]) - (x[1]*z[3]*y[1]) - \
    #                 (x[2]*z[2]*y[3]) + (x[3]*z[2]*y[1])

    #             self._mixed_eq_str += f"\n Quad({self.equation_variables[i]})= \t {c_2} {self.equation_variables[i]} ^2 + {c_1} {self.equation_variables[i]} + {c_0}"
    #             solutions=self.compute_quadratic_roots(c_2, c_1, c_0)


# def player_shape_to_intersection_shape(br_shape):
#     int_shape = None
#     if br_shape == BR_Shape.Constant:
#         int_shape = Intersection_Shape.Partial
#     elif br_shape == BR_Shape.Hyperbola:
#         int_shape = Intersection_Shape.GenHype
#     elif br_shape == BR_Shape.DegHyperbola:
#         int_shape = Intersection_Shape.DegHype
#     elif br_shape == BR_Shape.Line:
#         int_shape = Intersection_Shape.GenLine
#     return int_shape
