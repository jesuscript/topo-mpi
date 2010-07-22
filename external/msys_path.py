import sys
import os.path

target = sys.argv[1][:]
path1 = sys.argv[2]
path2 = sys.argv[3]

f = open(target,'w')
f.write("import site;site.addsitedir('%s');site.addsitedir('%s')\n"%(path1,path2))
f.close()

