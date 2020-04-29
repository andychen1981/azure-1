import os, sys
    
def direxist(dirname):
	return os.path.isdir(dirname) and os.path.exists(dirname)

def mkdir(dirname):
	if not direxist(dirname):
		os.mkdir(dirname)

def mkdirname(dirname):
	return dirname if dirname[-1] == '/' else dirname + '/'