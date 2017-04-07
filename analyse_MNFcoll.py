USAGE = 'python analyse_MNFcoll.py -i <file in (box)>'

import MN_Fcoll as mn
import sys, getopt

file_in_opt = ""

try:
    opts, args = getopt.getopt(sys.argv[1:], "h:i:")
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
if file_in_opt == "":
    print USAGE
    sys.exit()

mn_obj = mn.MN(filename=file_in_opt)
mn_obj.save_modes()
mn_obj.marginals()
