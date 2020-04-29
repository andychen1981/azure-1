#!/usr/bin/env python2.7

import os, sys
from subprocess import call
#from multiprocessing import Pool, Process
import multiprocessing as mp
#import python27utils.saw.loadscores as loadscores
import folderiter as folderiter
import mt as mt

#
# http://sebastianraschka.com/Articles/2014_multiprocessing.html
#

kNumThreads=4
script='copy1.sh'

def info(title):
    print(title)
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f2(name):
    info('function f')
    print('hello', name)

def onefile(filepath, context):
    context[filepath] = None        #value will be filled by the threadproc

# define a thread processing function
def threadproc(length, x, outputQ):
    #info(x)
    basename = os.path.basename(x)
    base, ext = os.path.splitext(basename)
    dstf = dstfolder + basename
    cmdline = "./%s %s %s" % (script, x, dstf)
    outputQ.put(dstf)       #this is owned by the main thread
    #print(cmdline)
    call(cmdline, shell=True)

# define a persistent thread processing function
# this is not really used but is an example of what a threadproc should look like
def threadprocPT(length, i, q, outputQ):
    #info(x)
    while not q.empty():
        item = q.get()
        #do_work(item)
        #print("thread(%d): " % i)
        print(item)
        basename = os.path.basename(item)
        base, ext = os.path.splitext(basename)
        dstf = dstfolder + basename
        cmdline = "./%s %s %s" % (script, item, dstf)
        #print(cmdline)
        call(cmdline, shell=True)

        #q.task_done()
        outputQ.put(item)

if __name__ == '__main__':
    srcfolder = 'src'
    dstfolder = 'dst'
 
    srcfolder = folderiter.addslash(srcfolder)
    dstfolder = folderiter.addslash(dstfolder)

    worktodo = dict()
    queue = mp.Queue(maxsize=len(worktodo))

    folderiter.folder_iter(srcfolder, onefile, worktodo, file_filter=folderiter.pngfile_filter)
    #input2scores = loadscores.buildinput2scores(srcfolder, dstfolder)
    print('# of workitems', len(worktodo))
    for i, key in enumerate(worktodo):
        queue.put(key)

    #workdone = mt.processbatches(queue, worktodo, threadproc, kNumThreads, False)
    workdone = mt.processbatchesPT(queue, threadprocPT, kNumThreads, False)

    print(workdone)

