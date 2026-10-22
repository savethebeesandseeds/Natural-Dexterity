"""Private scene storage helpers; no web or provider operations."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ordinary(path):
    path=Path(path)
    if any(part.is_symlink() for part in (path,*path.parents)):
        raise ValueError("Scene storage must not traverse symlinks.")
    return path


def read(path,limit):
    path=ordinary(path)
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(descriptor,"rb") as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=limit:
            raise ValueError("Scene source must be a bounded ordinary file.")
        data=stream.read(limit+1)
    if len(data)>limit:
        raise ValueError("Scene source exceeds its bound.")
    return data


def write(path,data):
    path=ordinary(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path,max(len(data),1))!=data:
            raise ValueError("An immutable scene artifact already exists with different bytes.")
        return
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=".scene-write-",delete=False) as stream:
            temporary=Path(stream.name);stream.write(data);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,path)
        except FileExistsError:
            if read(path,max(len(data),1))!=data:raise ValueError("Scene artifact conflicts with retained data.")
    finally:
        if temporary:temporary.unlink(missing_ok=True)


@contextmanager
def lease(path,*,exclusive=False):
    path=ordinary(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    with os.fdopen(descriptor,"r+b") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Scene operation lease must be an ordinary file.")
        try:fcntl.flock(stream,(fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError("A coordinated backup or scene operation is active; retry later.") from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)
