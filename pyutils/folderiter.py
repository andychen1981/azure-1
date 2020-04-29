import os, sys

# default folder filer is a pass-through:
def deffolder_filter(subdir): 
	subd = os.path.basename(subdir)
	return True if subd == '' else subd[0] != '.' and subd[0] != '..'

# default file filter - ignore '.xxx'
deffile_filter = lambda file: True if file == '' else file[0] != '.'
# selects .png files
pngfile_filter = lambda file: deffile_filter(file) and (os.path.splitext(file)[-1].lower() == '.png')
# selects .jpg files
jpgfile_filter = lambda file: deffile_filter(file) and (os.path.splitext(file)[-1].lower() == '.jpg')
# selects .npy files
npyfile_filter = lambda file: deffile_filter(file) and (os.path.splitext(file)[-1].lower() == '.npy')
# selects .wav files
wavfile_filter = lambda file: deffile_filter(file) and (os.path.splitext(file)[-1].lower() == '.wav')

def addslash(folder):
	return folder if folder[-1] == '/' else folder + '/'

def ensure_dir_exists(dir_name):
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
		
def folder_iter(
	rootdir,
	functor,
	context = None,
	folder_filter = deffolder_filter,
	file_filter   = deffile_filter,
	logging=True
):
	for subdir, dirs, files in os.walk(rootdir):
		#print("subdir %s" % subdir)
		if not deffolder_filter(subdir):
			continue

		if logging:
			print("subdir %s" % subdir)
		#print(os.path.dirname(subdir))
		for file in files:
			if (file_filter(file)):
				filepath = os.path.join(subdir, file)
				functor(filepath, context)
	return
