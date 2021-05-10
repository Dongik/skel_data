import pandas as pd


gn = 6
pn = 16
gn_n = gp + pn

sn = 51 # 17 * 3


# left gyro start column index
lgs = 1 

# left pressure start column index
lps = 7


# left gyro start column index
rgs = 23

# left pressure start column index
rps = 29

ss = 45


p_cols = np.r_[lps: lps + pn, rps: rps + pn]
s_cols = np.r_[ss: ss + sn]

gp_cols = np.r_[lgs:lgs + gp_n, rgs + gp_n]