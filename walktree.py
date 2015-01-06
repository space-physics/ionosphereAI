"""
Recursively loops for pattern-matching files, like GNU find
Michael Hirsch
Dec 2014
"""
from os import walk
from os.path import join,expanduser,isdir, isfile
from fnmatch import filter
#from stat import S_ISDIR, S_ISREG

def walktree(root,pat):
    root = expanduser(root)
    if isdir(root):
        found = []

        for top,dirs,files in walk(root):
            for f in filter(files,pat):
                found.append(join(top,f))

        # this is optional--I like to use None as a sentinal value
        if len(found)==0:
            found=None
    elif isfile(root):
        found = [root]
    else:
        raise NotImplementedError("is",root,"a file or directory?")

    return found


if __name__ == '__main__':
    from sys import argv
    if len(argv) != 3:
        exit('USAGE: walktree.py rootdir pattern')
    found = walktree(argv[1],argv[2])
    print(found)