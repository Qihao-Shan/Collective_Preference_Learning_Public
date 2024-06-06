import numpy as np
import matplotlib.pyplot as plt
import arena_class
import dm_objects


fig, axis = plt.subplots(2, 2)

Max_step = 240000
n_robot = 20
#arena_width = arena_class.compute_proportional_arena_width(n_robot)
s = arena_class.sites(noise_size=2)
dm = dm_objects.DM_object_fusion(n_robot, s)
#dm = dm_objects.DM_object_voting(n_robot, s, election_size=1)
a = arena_class.arena(dm, N=n_robot, axis=axis)
true_belief_mat = dm_objects.ranking_to_mat(7-s.site_quality_array)
print(true_belief_mat)

for i in range(Max_step):
    a.random_walk_mat()
    if i % 100 == 0:
        a.dm_object.make_decision(a.coo_array)
        #print(a.dm_object.ranking_array)
        #print(true_belief_mat)
        #print(a.dm_object.belief_mat[0, :, :])
        print(a.dm_object.compute_error(true_belief_mat))
    a.plot_arena(i)




