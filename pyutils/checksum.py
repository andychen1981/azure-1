import sys
from hashlib import md5
from zlib import crc32

def md5sum(filename):
    hash = md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * hash.block_size), b""):
            hash.update(chunk)
    return hash.hexdigest()

def crc32sum(filename):
    hash = 0
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1024*32), b""):
            hash = crc32(chunk, hash)
    return hash & 0xffffffff

def crc32str(ourstr):
    return crc32(outstr, 0)
    