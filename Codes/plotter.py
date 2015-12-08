#!/usr/bin/python

from matplotlib import pyplot as pp

pp.ion()

def gp_vs_svm():
    n = [100, 200, 300, 500, 1000, 1500, 2000 ]

    gp_d_m =  [ 0.33793593045644227, 0.30868032509805804, 0.29367738237107543, 0.277077135625457, 0.25989584878330485, 0.25405548599901806, 0.24955768654213006 ]
    svr_d_m = [ 0.3542099829961689, 0.3100033404397664, 0.29872494710118896, 0.2851612962458555, 0.26741686816089055, 0.2621969913907439, 0.258187518844308 ];

    pp.figure()

    pp.plot(n, gp_d_m, 'r')
    pp.plot(n, svr_d_m, 'b')
    pp.ylabel('$\Delta_{rt}^{95}$')
    pp.xlabel('Size of training data')
    pp.legend(['GP', 'SVR'])
    pp.title('GP vs SVR')
    pp.grid()
    pp.savefig('gp_vs_svr.pdf')
