USAGE = '[mpiexec -n 4] python run_MNFcoll.py -i <file in (box)> [-n <number of live points>] [-s <scatter value>] [--resume]'

import MN_Fcoll as mn
import sys, getopt

file_in_opt = ""
resume_opt = False
live_points_opt = 1000
scatter_opt = 1.

try:
    opts, args = getopt.getopt(sys.argv[1:], "h:i:n:s:", ["resume"])
except getopt.GetoptError:
    print USAGE
    sys.exit(2)
# print opts, args
for opt, arg in opts:
    if opt in ("-h", "--h", "--help"):
        print USAGE
        sys.exit()
    elif opt in ("-i"):
        file_in_opt = arg
    elif opt in ("--resume"):
        resume_opt = True
    elif opt in ("-n"):
        live_points_opt = int(arg)
    elif opt in ("-s"):
        scatter_opt = float(arg)
if file_in_opt == "":
    print USAGE
    sys.exit()

mn_obj = mn.MN(filename=file_in_opt)
mn_obj.run_sampling(marginals=False, n_points=live_points_opt, resume=resume_opt, scatter=scatter_opt)
