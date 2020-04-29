import os, sys
#from html import HTML
from utils.folderiter import folder_iter

rootdir    = './'

bestfolder = 'best20'
prevbestfolder = 'best2'
reffolder  = 'bell2014_densecrf'
#reffolder  = '/Users/mannyko/DevRoot/ResearchCode/extern/OpenSurf/iiw/iiw-decompositions/bell2014_densecrf'


# default folder filter is a pass-through:
def deffolder_filter(subdir): 
	subd = os.path.basename(subdir)
	if subd == '':
		return True

	if (subd[0] == '.') or (os.path.split(subdir)[-1] == 'shading'):
		return False
	return True 

pngfile_filter   = lambda file: os.path.splitext(file)[-1].lower() == '.png'

def writeHeader(f):
	message  = "<!DOCTYPE html>\n"
	message += "<html>\n"
	message += "<body>\n"
	f.write(message)
	return

def writerTrailer(f):
	message  = "</body>\n"
	message += "</html>\n"
	f.write(message)
	return

def writeTableHeader(f):
	message = "<table style=\"width:100%\">\n";
	f.write(message)
	return

def writeTableTrailer(f):
	message = "</table>\n";
	f.write(message)
	return

def getMatchingRef(filepath, reffolder):
	fn = os.path.basename(filepath)
	barename, ext = os.path.splitext(fn)
	return reffolder + '/' + barename + ext

def getShadingFile(filepath, shadingfolder):
	fn = os.path.basename(filepath)
	barename, ext = os.path.splitext(fn)
	shadingf = barename[0:-2] + '-s'
	return shadingfolder + '/' + shadingf + ext

def oneimg(filepath):
	f.write("<td><img src=")
	f.write('"')
	f.write(filepath)
	f.write('"')
	str = ' style="width:200px;\"'
	f.write('></td>\n')

def onefilename(f, filepath):
	f.write('<td>')
	f.write(filepath)
	f.write('</td>')

def perfile(
	filepath
):
	print filepath
	#one row of images
	f.write('<tr align = \"center\">\n')

	oneimg(filepath)

	prevbest = getMatchingRef(filepath, prevbestfolder)
	print('prevbest %s' % prevbest)
	oneimg(prevbest)

	reffile = getMatchingRef(filepath, reffolder)
	oneimg(reffile)

	shadingf = getShadingFile(filepath, bestfolder + '/shading')
	oneimg(shadingf)

	prevshadingf = getShadingFile(filepath, prevbestfolder + '/shading')
	oneimg(prevshadingf)

	refshading = getMatchingRef(shadingf, reffolder)
	oneimg(refshading)
	f.write('</tr>\n')

	# 2nd row for filename and distance value
	f.write('<tr align = \"center\">\n')
	onefilename(f, filepath)
	onefilename(f, prevbest)
	onefilename(f, reffile)

	onefilename(f, shadingf)
	onefilename(f, prevshadingf)
	onefilename(f, refshading)

	f.write('</tr>\n')

	return


# main()
if __name__ == '__main__':
	f = open('bestvizX.html','w')
	writeHeader(f)
	writeTableHeader(f)

	folder_iter(rootdir + bestfolder, perfile, deffolder_filter, pngfile_filter)

	writeTableTrailer(f)
	writerTrailer(f)
	f.close()
