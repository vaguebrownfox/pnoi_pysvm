# import libraries
import scipy.io
import scipy.stats as stats
import numpy as np
import random

# read mat files
pat = scipy.io.matlab.loadmat('./patient_label_0.mat')
con = scipy.io.matlab.loadmat('./control_label_1.mat')

# labels
lb_pat = 0
lb_con = 1

# constants
N_MFCC_COEFFS = 12
N_STAT_OP = 6
MODE_DCML = 3
ft_ncols = N_MFCC_COEFFS * N_STAT_OP + 1

# fetch mfcc matrices from mat file (3 per file)
p_b1 = np.array(pat.get('b1'))
p_b2 = np.array(pat.get('b2'))
p_b3 = np.array(pat.get('b3'))

c_b1 = np.array(con.get('b1'))
c_b2 = np.array(con.get('b2'))
c_b3 = np.array(con.get('b3'))

# labeled data
data_label = [(p_b1, lb_pat), (p_b2, lb_pat), (p_b3, lb_pat),
              (c_b1, lb_con), (c_b2, lb_con), (c_b3, lb_con)]


def feature_vec(p_b, label):
    # ''' compute feature vector from mfcc matrix (label appended) '''
    p_b_mean = np.mean(p_b, axis=0)
    p_b_median = np.median(p_b, axis=0)
    p_b_std = np.std(p_b, axis=0)
    p_b_var = np.var(p_b, axis=0)
    p_b_rms = np.sqrt(np.mean(np.square(p_b), axis=0))
    p_b_mode = stats.mode(p_b.round(decimals=MODE_DCML), axis=0)[0][0]
    f_vec = np.concatenate(
        [p_b_mean, p_b_median, p_b_std, p_b_var, p_b_rms, p_b_mode, [label]])

    return f_vec.reshape(1, N_MFCC_COEFFS * N_STAT_OP + 1)


# feature matrix with label
fm = np.zeros(ft_ncols).reshape(1, ft_ncols)
random.shuffle(data_label)
for d, l in data_label:
    # loop through data to construct feature matrix
    fv = feature_vec(d, l)
    fm = np.concatenate([fm, fv])

# write feature matrix to csv file
np.savetxt('features_svm.csv', fm, delimiter=',')
