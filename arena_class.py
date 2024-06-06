import random
import numpy as np
import math
import matplotlib.pyplot as plt
import dm_objects


TIME_STEP = 0.01
TSPF = 100
#EXP_SCALE = 0.02
SAMPLE_NUM = 10
#Site_interaction_rate = 0.1


# COST_FACTOR = -1.

class epuck:
    def __init__(self):
        # constants
        self.vl = 0.16
        self.va = 0.75
        self.r = 0.035
        #self.comm_dist = 0.5  # 0.5
        # movement
        self.dir = random.random() * 2 * math.pi
        self.dir_v = np.array([math.sin(self.dir), math.cos(self.dir)])
        self.turn_dir = int(random.random() * 2) * 2 - 1
        self.walk_state = int(random.random() * 2)
        self.walk_timer = 0


def compute_proportional_arena_width(num_robots):
    return np.sqrt(num_robots/30 * 9)


class sites:
    def __init__(self, noise_size, evidence_rate=1, arena_width=3):
        self.site_index_array = np.array(range(8))
        self.site_quality_array = np.random.permutation(np.array(range(8)))  #np.random.uniform(0, 10, 8)
        self.site_coo_array = np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
                                        [1.5, 0.5],              [1.5, 2.5],
                                        [2.5, 0.5], [2.5, 1.5], [2.5, 2.5]]) * arena_width / 3
        self.site_r = 0.3 * arena_width / 3
        self.noise_size = noise_size
        self.evidence_rate = evidence_rate

    def obtain_quality(self, coo_array):
        (n_robot, _) = np.shape(coo_array)
        (n_site, _) = np.shape(self.site_coo_array)
        coo_mat = np.tile(coo_array, (n_site, 1, 1))
        site_coo_mat = np.tile(self.site_coo_array, (n_robot, 1, 1)).transpose((1, 0, 2))
        dist_mat = np.sqrt(np.sum((coo_mat-site_coo_mat)**2, axis=2))
        closest_site_array = np.argmin(dist_mat, axis=0)
        ra = np.array(range(n_robot))
        site_quality_mat = np.tile(self.site_quality_array, (n_robot, 1))
        index_mat = np.vstack((ra, closest_site_array))
        dist_array = np.min(dist_mat, axis=0)
        quality_array = site_quality_mat[tuple(index_mat)]
        closest_site_array[dist_array > self.site_r] = -1
        #if np.any(self.site_index_array == -1):
        #    closest_site_array[closest_site_array == 0] = -1
        #    closest_site_array[closest_site_array == 2] = -1
        #    closest_site_array[closest_site_array == 5] = -1
        #    closest_site_array[closest_site_array == 7] = -1
        # interact probability
        r_interact = np.random.uniform(size=n_robot)
        closest_site_array[r_interact > self.evidence_rate] = -1
        #print(quality_array)
        quality_array = np.random.normal(quality_array, self.noise_size)
        return closest_site_array, quality_array


class arena:
    def __init__(self, dm_object, N=20, dim=np.array([3, 3]), axis=None):
        # initialise arena
        self.n_robot = N
        self.robot_spec = epuck()
        # initialise agents
        self.coo_array = np.array([]).reshape([0, 2])
        self.n = float(N)
        self.dim = dim
        for i in range(N):
            coo = np.array([random.random(), random.random()] * self.dim)
            #self.robots.append(epuck())
            while self.collision_detect(self.coo_array, coo):
                coo = np.array([random.random(), random.random()] * self.dim)
                #print('new position', i, coo)
            self.coo_array = np.vstack((self.coo_array, coo))
        self.axis = axis
        self.dm_object = dm_object
        #self.dm_object.comm_dist = self.robot_spec.comm_dist
        self.dir_array = np.random.rand(N) * 2 * math.pi
        self.dir_v_array = np.vstack((np.sin(self.dir_array), np.cos(self.dir_array))).T
        self.turn_dir_array = np.floor(np.random.rand(N) * 2) * 2 - 1  # randomly -1 or +1
        self.walk_state_array = np.floor(np.random.rand(N) * 2)  # randomly 0 or 1
        self.walk_timer_array = np.zeros(N)

    def oob(self, coo):
        # out of bound
        if self.robot_spec.r < coo[0] < self.dim[0] - self.robot_spec.r \
                and self.robot_spec.r < coo[1] < self.dim[1] - self.robot_spec.r:
            return False
        else:
            #print('oob ', coo)
            return True

    def collision_detect(self, coo_array, new_coo):
        # check if new_coo clip with any old coo, or oob
        if self.oob(new_coo):
            return True
        elif coo_array.shape[0] == 0:
            return False
        else:
            dist_array = np.sqrt(np.sum((coo_array - new_coo) ** 2, axis=1))
            if np.min(dist_array) < 2 * self.robot_spec.r:
                #print(dist_array)
                #print('collision ')
                return True
            else:
                return False

    def collision_detect_2(self):  # output array indicating collision
        new_coo_array = self.coo_array + self.dir_v_array * self.robot_spec.vl * TIME_STEP * 10
        new_coo_mat = np.tile(new_coo_array, (int(self.n), 1, 1)).transpose((1, 0, 2))
        coo_mat = np.tile(self.coo_array, (int(self.n), 1, 1))
        dist_mat = np.sqrt(np.sum((new_coo_mat - coo_mat) ** 2, axis=2))
        dist_mat += np.identity(dist_mat.shape[0]) * 100
        collision_mat = np.zeros_like(dist_mat)
        collision_mat[dist_mat < 2 * self.robot_spec.r] = 1
        collision_array = np.sum(collision_mat, axis=1)
        collision_array[collision_array > 1] = 1
        oob_arr = self.oob_array(new_coo_array)
        collision_array[oob_arr] = 1
        return collision_array

    def oob_array(self, coo_array):
        horizontal = np.logical_or(coo_array[:, 0] < self.robot_spec.r,
                                   coo_array[:, 0] > self.dim[0] - self.robot_spec.r)
        vertical = np.logical_or(coo_array[:, 1] < self.robot_spec.r,
                                 coo_array[:, 1] > self.dim[1] - self.robot_spec.r)
        return np.logical_or(horizontal, vertical)

    def random_walk_mat(self):
        self.walk_timer_array -= 1
        # for state=0
        collision_array = self.collision_detect_2()
        move_array = collision_array + self.walk_state_array
        self.coo_array[move_array == 0, :] += self.dir_v_array[move_array == 0, :] * self.robot_spec.vl * TIME_STEP
        time_out_array_0 = np.logical_and(self.walk_timer_array < 0, self.walk_state_array == 0)
        switch_array_0 = np.logical_or(time_out_array_0, np.logical_and(collision_array == 1,
                                                                        self.walk_state_array == 0))
        switch_array_0 = np.logical_or(switch_array_0, np.logical_and(self.walk_timer_array < 0, collision_array == 1))
        self.walk_state_array[switch_array_0] = 1
        self.walk_timer_array[switch_array_0] = np.random.rand(
            self.walk_state_array[switch_array_0].size) * 4.5 / TIME_STEP
        self.turn_dir_array[switch_array_0] = np.floor(
            np.random.rand(self.walk_state_array[switch_array_0].size) * 2) * 2 - 1
        # for state=1
        self.dir_array[self.walk_state_array == 1] += \
            self.turn_dir_array[self.walk_state_array == 1] * self.robot_spec.va * TIME_STEP
        self.dir_v_array = np.vstack((np.sin(self.dir_array), np.cos(self.dir_array))).T
        switch_array_1 = np.logical_and(self.walk_state_array == 1, self.walk_timer_array < 0)
        self.walk_state_array[switch_array_1] = 0
        self.walk_timer_array[switch_array_1] = np.random.rand(
            self.walk_state_array[switch_array_1].size) * 4.5 / TIME_STEP

    def plot_arena(self, t_step):
        if t_step % TSPF == 0:
            self.axis[0, 0].cla()  # site and robots
            self.axis[0, 1].cla()
            self.axis[1, 0].cla()
            self.axis[1, 1].cla()

            self.axis[0, 0].set_title('timestep '+str(t_step))
            for i in range(np.size(self.dm_object.sites.site_quality_array)):
                circle = plt.Circle((self.dm_object.sites.site_coo_array[i, 0], self.dm_object.sites.site_coo_array[i, 1]), self.dm_object.sites.site_r, color='k', alpha=(self.dm_object.sites.site_quality_array[i]+1)/9)
                self.axis[0, 0].add_artist(circle)

            for i in range(np.shape(self.coo_array)[0]):
                circle = plt.Circle((self.coo_array[i, 0], self.coo_array[i, 1]), self.robot_spec.r, color='r', fill=False)
                self.axis[0, 0].add_artist(circle)
                self.axis[0, 0].plot(np.array([self.coo_array[i, 0], self.coo_array[i, 0]+self.dir_v_array[i, 0]*0.05]), np.array([self.coo_array[i, 1], self.coo_array[i, 1]+self.dir_v_array[i, 1]*0.05]),'b')
            self.axis[0, 0].plot(self.coo_array[:, 0],
                              self.coo_array[:, 1], 'ro', markersize=3)
            #self.axis[0, 1].plot(self.dm_object.decision_array, 'r*')
            #self.axis[0, 1].set(xlim=(-1, 40), ylim=(-1, 8))
            self.axis[1, 0].set_title('True ranking')
            self.axis[1, 0].plot(7 - self.dm_object.sites.site_quality_array, 'b*')
            self.axis[0, 0].set(xlim=(0, self.dim[0]), ylim=(0, self.dim[1]))
            self.axis[0, 0].set_aspect('equal', adjustable='box')

            # strategy specific monitoring, comment out if error
            if self.dm_object.dm_type == 'voting':
                for i in range(self.n_robot):
                    self.axis[1, 1].plot(self.dm_object.ranking_array[i, :], 'g*', alpha=0.05)
                self.axis[1, 1].set(xlim=(-1, 8), ylim=(-2, 8))
                self.axis[1, 1].set_title('Computed ranking')
            elif self.dm_object.dm_type == 'fusion':
                true_belief_mat = dm_objects.ranking_to_mat(7 - self.dm_object.sites.site_quality_array)
                for i in range(8):
                    for j in range(8):
                        self.axis[1, 1].plot(i, j, 'g*', alpha=np.sum(np.abs(self.dm_object.belief_mat[:, i, j] - true_belief_mat[i, j]))/60)
                self.axis[1, 1].set(xlim=(-1, 8), ylim=(-1, 8))
                self.axis[1, 1].set_title('Belief matrix')
            plt.draw()
            plt.pause(0.001)
        else:
            pass


