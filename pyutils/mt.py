from multiprocessing import Pool, Process
import os
import multiprocessing as mp
from subprocess import call
#import psutil

#
# http://sebastianraschka.com/Articles/2014_multiprocessing.html
# https://pymotw.com/2/multiprocessing/basics.html

kNumThreads=10
#output = mp.Queue()

def info(title):
    print(title)
    print('module name: %s' % __name__)

    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())

# define a thread processing function
# this is not really used by act as a example of what a threadproc should look like
def threadproc(length, x, outputQ):
    #info(x)
    outputQ.put(x)                                  #update our queue
    basename = os.path.basename(x)
    cmdline = "./%s %s" % (script, basename)        # totally not general - assumes a single argument for script
    call(cmdline, shell=True)

def processbatches(
    outputQ,                #output queue (optional) - usually created in the main thread
    work,                   #work specified by a dict() key set 
    threadproc,             #per-thread handler
    numcores=kNumThreads,
    logging=True,
    minibatch=True
):
    if logging:
        info('processbatches')
    #p = Pool(5)
    #print(p.map(f, [1, 2, 3]))
    #batch = input2scores        #TODO: partition into batch based on 'numcores' and len(input2scores)
    workdone = list()

    ouritems = work.items()
    start = 0
    end   = kNumThreads if minibatch else len(ouritems)
    batch = dict(ouritems[start:end])
    activeprocs = 0

    while len(batch) != 0:
        # Setup a list of processes that we want to run
        processes = [mp.Process(target=threadproc, args=(5, key, outputQ)) for key in batch]  #range(4)

        # Run processes
        for p in processes:
            p.start()
            activeprocs += 1

        # wait for each process
        for p in processes:
            p.join()
            activeprocs -= 1
            results = [outputQ.get()]
            workdone += results

        # Get process results from the output queue
        #results = [outputQ.get() for p in processes]
        #workdone += results

        if logging:
            print("batch", results)

        start += kNumThreads
        end   += kNumThreads        #implicitly assumes each workitem cost the same - use CDF later if load is highly variable
        end = min(len(ouritems), end)   # this is not really needed since the slicing handles it
        batch = dict(ouritems[start:end])

    return workdone
    #pool = Pool(processes=4)
    #pool.map(f, ['file1','file2','file3','file4'])

    #for i in pool.imap_unordered(f, ['file1','file2','file3','file4']):
        #i
 
    #p = Process(target=f2, args=('bob',))
    #p.start()
    #p.join()


# define a persistent thread processing function
# this is not really used by act as a example of what a threadproc should look like
def threadprocPT(length, i, q, outputQ):
    #info(x)
    while not q.empty():
        item = q.get()
        #do_work(item)
        #print item
        #q.task_done()
        #basename = os.path.basename(item)
        #cmdline = "./%s %s" % (script, basename)        # totally not general - assumes a single argument for script
        #call(cmdline, shell=True)
        outputQ.put(item)

def processbatchesPT(
    inputQ,                 #input queue - usually created in the main thread
    threadproc,             #per-thread handler
    numcores=kNumThreads,
    logging=True
):
    if logging:
        info('processbatchesPT')    #persistent thread support
  
    outputQ = mp.Queue()
 
    #ouritems = len(inputQ)
    processes = list()

    # Setup a list of processes that we want to run
    for i in range(numcores):
        processes.append(mp.Process(target=threadproc, args=(5, i, inputQ, outputQ)))

    # Run processes
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    #inputQ.join()
 
    if logging:
        print("batch", results)

    return outputQ