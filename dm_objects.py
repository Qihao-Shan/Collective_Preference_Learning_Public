import numpy as np
import scipy.stats
import random

COMM_DIST = 0.5


def check_dominant_decision(dm_object, threshold):
    # print(dm_object.decision_array)
    decisions_tally = np.bincount(dm_object.decision_array)
    dominant_decision = np.argmax(decisions_tally)
    consensus_number = np.max(decisions_tally)
    if consensus_number > dm_object.n * threshold:
        # print('consensus made ', consensus_number, dm_object.decision_array)
        return dominant_decision
    else:
        return -1


def compute_convergence_error(belief_mat):
    (n, _, _) = np.shape(belief_mat)
    belief_mat_full = np.tile(belief_mat, (n, 1, 1, 1))
    belief_mat_full_ = np.transpose(belief_mat_full, (1, 0, 2, 3))
    error = np.sum(np.abs(belief_mat_full-belief_mat_full_)) / float(n*(n-1))
    return error


def ranking_to_mat(ranking):
    n = np.size(ranking)
    belief_mat = np.zeros((n, n))
    for i in range(np.size(ranking)):
        if ranking[i] != -1:
            belief_mat[i, np.logical_and(ranking < ranking[i], ranking != -1)] = -1
            belief_mat[i, ranking > ranking[i]] = 1
            belief_mat[np.logical_and(ranking < ranking[i], ranking != -1), i] = 1
            belief_mat[ranking > ranking[i], i] = -1
    return belief_mat


class DM_object:
    def __init__(self, n, sites):
        self.n = n
        self.site_quality_array = -np.ones((n, 2)) * 100
        self.site_index_array = -np.ones((n, 2))
        self.neighbour_mat = np.zeros((n, n))
        self.sites = sites
        self.noise_size = 0

    def compute_neighbour(self, coo_array):
        neighbour_mat = np.zeros((self.n, self.n))
        coo_mat = np.tile(coo_array, (self.n, 1, 1))
        coo_mat_ = coo_mat.transpose((1, 0, 2))
        dist_mat = np.sqrt(np.sum((coo_mat - coo_mat_) ** 2, axis=2)) + np.identity(self.n) * 100
        neighbour_mat[dist_mat < COMM_DIST] = 1
        return neighbour_mat


class DM_object_voting(DM_object):
    def __init__(self, n, sites, election_size):
        super().__init__(n, sites)
        self.dm_type = 'voting'
        self.n_robots = n
        self.n_sites = np.size(sites.site_index_array)
        self.ballot_record = -np.ones((n, election_size, sites.site_index_array.size))
        self.ranking_array = -np.ones((n, sites.site_index_array.size))
        self.voter_record = -np.ones((n, election_size))

    def update_ranking(self, i, site_index, quality):
        # update internal record of sites using id and quality inputs, if identical id, replace identical, otherwise replace random
        if self.site_index_array[i, 0] == site_index or self.site_index_array[i, 0] == -1:
            replace_index = 0
        elif self.site_index_array[i, 1] == site_index or self.site_index_array[i, 1] == -1:
            replace_index = 1
        else:
            replace_index = np.random.choice([0, 1])
        self.site_index_array[i, replace_index] = site_index
        self.site_quality_array[i, replace_index] = quality
        # update ranking, if ranking does not comply with comparison, switch position
        # if one of the sites is not in the ranking, add to random correct position
        ranking_array_ = self.ranking_array[i, :]
        if not np.any(self.site_index_array[i, :] == -1):
            ranking_0 = ranking_array_[int(self.site_index_array[i, 0])]
            ranking_1 = ranking_array_[int(self.site_index_array[i, 1])]
            if self.site_quality_array[i, 0] < self.site_quality_array[i, 1]:  # make sure ra_0 is the bigger term
                ranking_ = ranking_0
                ranking_0 = ranking_1
                ranking_1 = ranking_
                index_ = self.site_index_array[i, 0]
                self.site_index_array[i, 0] = self.site_index_array[i, 1]
                self.site_index_array[i, 1] = index_
                quality_ = self.site_quality_array[i, 0]
                self.site_quality_array[i, 0] = self.site_quality_array[i, 1]
                self.site_quality_array[i, 1] = quality_
            if ranking_0 < 0:
                if ranking_1 < 0:
                    # both not found
                    if np.max(ranking_array_) < 0:
                        # no elements in list
                        ranking_array_[int(self.site_index_array[i, 0])] = 0
                        ranking_array_[int(self.site_index_array[i, 1])] = 1
                    else:
                        # there are elements in ranking, but none of the two found sites
                        index_range_0 = np.array(range(int(np.max(ranking_array_)+2)))
                        index_0 = np.random.choice(index_range_0)
                        ranking_array_[ranking_array_ >= index_0] += 1
                        ranking_array_[int(self.site_index_array[i, 0])] = index_0
                        index_range_1 = np.array(range(int(np.max(ranking_array_)+2)))
                        index_range_1 = index_range_1[index_range_1 > index_0]
                        index_1 = np.random.choice(index_range_1)
                        ranking_array_[ranking_array_ >= index_1] += 1
                        ranking_array_[int(self.site_index_array[i, 1])] = index_1
                else:
                    # ra_1 found
                    index_range_0 = np.array(range(int(ranking_1 + 1)))
                    index_0 = np.random.choice(index_range_0)
                    ranking_array_[ranking_array_ >= index_0] += 1
                    ranking_array_[int(self.site_index_array[i, 0])] = index_0
            else:
                if ranking_1 < 0:
                    # ra_0 found
                    index_range_1 = np.array(range(int(np.max(ranking_array_) + 1)))
                    index_range_1 = index_range_1[index_range_1 >= ranking_0]
                    index_1 = np.random.choice(index_range_1)
                    ranking_array_[ranking_array_ >= index_1] += 1
                    ranking_array_[int(self.site_index_array[i, 1])] = index_1
                else:
                    # both found, remove random one if ordering not correct
                    if ranking_0 > ranking_1:
                        #position_range = np.array([self.site_index_array[i, 0], self.site_index_array[i, 1]])
                        #position = np.random.choice(position_range)
                        #ranking_array_[ranking_array_ > ranking_array_[int(position)]] -= 1
                        #ranking_array_[int(position)] = -1
                        # switch
                        ranking_array_[int(self.site_index_array[i, 0])] = ranking_1
                        ranking_array_[int(self.site_index_array[i, 1])] = ranking_0
        return ranking_array_

    def add_ballot(self, voter_record, ballot_record, j, ballot):
        # add ballot to ballot record, if identical id, replace identical, otherwise add
        ra = np.array(range(np.size(voter_record)))
        r_v = ra[voter_record == j]
        if np.size(r_v) == 0:
            voter_record_ = voter_record[voter_record >= 0]
            voter_record[np.size(voter_record_)] = j
            ballot_record[np.size(voter_record_), :] = ballot
        else:
            ballot_record[r_v[0], :] = ballot
        return voter_record, ballot_record

    def election(self, i, ballot_record):
        # flush ballot record and output new ranking
        ballot_record_ = np.vstack((ballot_record, self.ranking_array[i, :]))
        max_row = np.max(ballot_record_, axis=0)
        (M, N) = np.shape(ballot_record_)
        #random_mat = np.random.uniform(0, 1, (M, N))
        mult_mat = np.tile(np.max(ballot_record_, axis=1), (N, 1)).T + 1
        random_mat = mult_mat * 0.5
        ballot_record_padded = random_mat
        ballot_record_padded[ballot_record_ >= 0] = ballot_record_[ballot_record_ >= 0]
        ballot_record_padded[:, max_row < 0] = -1
        tallied_array = np.sum(ballot_record_padded, axis=0)  # normal borda count
        #ballot_record_padded[ballot_record_padded >= 0] += 1  # dowdall system
        #tallied_array = np.sum(1/ballot_record_padded, axis=0)
        # print(tallied_array)
        tallied_array_pos = tallied_array[tallied_array >= 0]
        # print(tallied_array_pos)
        sort_ind = -np.ones_like(tallied_array_pos)
        ra = np.array(range(np.size(tallied_array_pos)))
        while np.size(sort_ind[sort_ind < 0]) > 0 and np.size(tallied_array_pos) > 0:
            minimum_term = np.min(tallied_array_pos)  # normal min, dowdall max
            ind = np.random.choice(ra[tallied_array_pos == minimum_term])
            tallied_array_pos[ind] = +1000  # normal +, dowdall -
            sort_ind[ind] = np.max(sort_ind) + 1
        new_ranking = -np.ones(N)
        new_ranking[tallied_array >= 0] = sort_ind
        # print(new_ranking)
        return new_ranking

    def make_decision(self, coo_array):
        (input_site_index_array, input_site_quality_array) = self.sites.obtain_quality(coo_array)
        neighbour_mat = self.compute_neighbour(coo_array)
        # neighbour_mat += np.identity(neighbour_mat.shape[0])
        for i in range(self.n):
            if input_site_index_array[i] >= 0:
                self.ranking_array[i, :] = self.update_ranking(i, input_site_index_array[i],
                                                               input_site_quality_array[i])
            potential_neighbour_list = np.array(range(self.n))[neighbour_mat[i, :] == 1]
            if np.size(potential_neighbour_list) > 0 > input_site_index_array[i]:
                neighbour_index = np.random.choice(potential_neighbour_list)
                (self.voter_record[i, :], self.ballot_record[i, :, :]) = self.add_ballot(self.voter_record[i, :],
                                                                                         self.ballot_record[i, :, :],
                                                                                         neighbour_index,
                                                                                         self.ranking_array[
                                                                                             neighbour_index])
                voter_record_ = self.voter_record[i, :]
                if np.size(voter_record_[voter_record_ < 0]) == 0:
                    self.ranking_array[i, :] = self.election(i, self.ballot_record[i, :, :])
                    self.ranking_array[i, :] = self.update_ranking(i, self.site_index_array[i, 0],
                                                                   self.site_quality_array[i, 0])
                    self.voter_record[i, :] = -1
                    self.ballot_record[i, :, :] = -1

    def compute_error(self, true_belief):
        error = 0
        belief_mat = np.zeros((self.n_robots, self.n_sites, self.n_sites))
        for i in range(self.n):
            belief = ranking_to_mat(self.ranking_array[i, :])
            belief_mat[i, :, :] = belief
            error += np.sum(np.abs(belief - true_belief))
        error_conv = compute_convergence_error(belief_mat)
        return error/float(self.n), error_conv


class DM_object_fusion(DM_object):
    def __init__(self, n, sites, transitivity=True):
        super().__init__(n, sites)
        self.n_robots = n
        self.n_sites = np.size(sites.site_index_array)
        self.belief_mat = np.zeros((self.n_robots, self.n_sites, self.n_sites))
        self.transitivity = transitivity
        self.dm_type = 'fusion'

    def update_belief(self, i, site_index, site_quality):
        # update internal record of sites using id and quality inputs, if identical id, replace identical, otherwise replace random
        if self.site_index_array[i, 0] == site_index or self.site_index_array[i, 0] == -1:
            replace_index = 0
        elif self.site_index_array[i, 1] == site_index or self.site_index_array[i, 1] == -1:
            replace_index = 1
        else:
            replace_index = np.random.choice([0, 1])
        self.site_index_array[i, replace_index] = site_index
        self.site_quality_array[i, replace_index] = site_quality
        # update belief mat
        belief_mat = self.belief_mat[i, :, :]
        if not np.any(self.site_index_array[i, :] == -1):
            if self.site_quality_array[i, 0] < self.site_quality_array[i, 1]:  # make sure site 0 is the bigger term
                index_ = self.site_index_array[i, 0]
                self.site_index_array[i, 0] = self.site_index_array[i, 1]
                self.site_index_array[i, 1] = index_
                quality_ = self.site_quality_array[i, 0]
                self.site_quality_array[i, 0] = self.site_quality_array[i, 1]
                self.site_quality_array[i, 1] = quality_
            belief_mat[int(self.site_index_array[i, 0]), int(self.site_index_array[i, 1])] = 1
            belief_mat[int(self.site_index_array[i, 1]), int(self.site_index_array[i, 0])] = -1
        return belief_mat

    def belief_fusion(self, i, neighbour_belief_mat):
        belief_mat = self.belief_mat[i, :, :]
        belief_mat += neighbour_belief_mat
        belief_mat[belief_mat > 0] = 1
        belief_mat[belief_mat < 0] = -1
        return belief_mat

    def preserve_transitivity(self, i):
        belief_mat = self.belief_mat[i, :, :]
        ra = np.array(range(np.size(self.sites.site_index_array)))
        for j in range(np.size(self.sites.site_index_array)):
            belief_slice = belief_mat[j, :]
            bigger_ind_array = ra[belief_slice == 1]
            smaller_ind_array = ra[belief_slice == -1]
            if np.size(bigger_ind_array) > 0 and np.size(smaller_ind_array) > 0:
                for b_ind in bigger_ind_array:
                    for s_ind in smaller_ind_array:
                        if belief_mat[b_ind, s_ind] == 0:
                            belief_mat[b_ind, s_ind] = -1
                            belief_mat[s_ind, b_ind] = 1
        return belief_mat

    def make_decision(self, coo_array):
        (input_site_index_array, input_site_quality_array) = self.sites.obtain_quality(coo_array)
        neighbour_mat = self.compute_neighbour(coo_array)
        for i in range(self.n):
            if input_site_index_array[i] >= 0:
                self.belief_mat[i, :, :] = self.update_belief(i, input_site_index_array[i], input_site_quality_array[i])
            potential_neighbour_list = np.array(range(self.n))[neighbour_mat[i, :] == 1]
            if np.size(potential_neighbour_list) > 0 > input_site_index_array[i]:
                neighbour_index = np.random.choice(potential_neighbour_list)
                self.belief_mat[i, :, :] = self.belief_fusion(i, self.belief_mat[neighbour_index, :, :])
            if self.transitivity:
                self.belief_mat[i, :, :] = self.preserve_transitivity(i)

    def compute_error(self, true_belief):
        full_true_belief_mat = np.tile(true_belief, (self.n_robots, 1, 1))
        error = np.sum(np.abs(full_true_belief_mat - self.belief_mat))
        error_conv = compute_convergence_error(self.belief_mat)
        return error/float(self.n), error_conv



