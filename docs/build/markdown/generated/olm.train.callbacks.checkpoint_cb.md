# olm.train.callbacks.checkpoint_cb

Checkpoint callback for saving model checkpoints during training.

### Classes

| [`CheckpointCallback`](#olm.train.callbacks.checkpoint_cb.CheckpointCallback)([checkpoint_dir, ...])   | Callback to save model checkpoints at specified intervals.   |
|--------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|

### *class* olm.train.callbacks.checkpoint_cb.CheckpointCallback(checkpoint_dir: str = 'checkpoints', save_every: int = 1000, keep_last_n: int = 5, save_best: bool = True)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to save model checkpoints at specified intervals.

* **Parameters:**
  * **checkpoint_dir** – Directory to save checkpoints.
  * **save_every** – Save checkpoint every N steps.
  * **keep_last_n** – Keep only the last N checkpoints.
  * **save_best** – Whether to save the best model based on validation loss.

#### on_step_end(trainer, step: int, loss: float) → None

Save checkpoint after each optimization step if needed.

### *class* olm.train.callbacks.checkpoint_cb.Path(\*args, \*\*kwargs)

Bases: `PurePath`

PurePath subclass that can make system calls.

Path represents a filesystem path but unlike PurePath, also offers
methods to do system calls on path objects. Depending on your system,
instantiating a Path will return either a PosixPath or a WindowsPath
object. You can also instantiate a PosixPath or WindowsPath directly,
but cannot instantiate a WindowsPath on a POSIX system or vice versa.

#### absolute()

Return an absolute version of this path
No normalization or symlink resolution is performed.

Use resolve() to resolve symlinks and remove ‘..’ segments.

#### as_uri()

Return the path as a URI.

#### chmod(mode, , follow_symlinks=True)

Change the permissions of the path, like os.chmod().

#### copy(target, \*\*kwargs)

Recursively copy this file or directory tree to the given destination.

#### copy_into(target_dir, \*\*kwargs)

Copy this file or directory tree into the given existing directory.

#### *classmethod* cwd()

Return a new path pointing to the current working directory.

#### exists(, follow_symlinks=True)

Whether this path exists.

This method normally follows symlinks; to check whether a symlink exists,
add the argument follow_symlinks=False.

#### expanduser()

Return a new path with expanded ~ and ~user constructs
(as returned by os.path.expanduser)

#### *classmethod* from_uri(uri)

Return a new path from the given ‘file’ URI.

#### glob(pattern, , case_sensitive=None, recurse_symlinks=False)

Iterate over this subtree and yield all existing files (of any
kind, including directories) matching the given relative pattern.

#### group(, follow_symlinks=True)

Return the group name of the file gid.

#### hardlink_to(target)

Make this path a hard link pointing to the same file as *target*.

Note the order of arguments (self, target) is the reverse of os.link’s.

#### *classmethod* home()

Return a new path pointing to expanduser(‘~’).

#### *property* info

A PathInfo object that exposes the file type and other file attributes
of this path.

#### is_block_device()

Whether this path is a block device.

#### is_char_device()

Whether this path is a character device.

#### is_dir(, follow_symlinks=True)

Whether this path is a directory.

#### is_fifo()

Whether this path is a FIFO.

#### is_file(, follow_symlinks=True)

Whether this path is a regular file (also True for symlinks pointing
to regular files).

#### is_junction()

Whether this path is a junction.

#### is_mount()

Check if this path is a mount point

#### is_socket()

Whether this path is a socket.

#### is_symlink()

Whether this path is a symbolic link.

#### iterdir()

Yield path objects of the directory contents.

The children are yielded in arbitrary order, and the
special entries ‘.’ and ‘..’ are not included.

#### lchmod(mode)

Like chmod(), except if the path points to a symlink, the symlink’s
permissions are changed, rather than its target’s.

#### lstat()

Like stat(), except if the path points to a symlink, the symlink’s
status information is returned, rather than its target’s.

#### mkdir(mode=511, parents=False, exist_ok=False)

Create a new directory at this given path.

#### move(target)

Recursively move this file or directory tree to the given destination.

#### move_into(target_dir)

Move this file or directory tree into the given existing directory.

#### open(mode='r', buffering=-1, encoding=None, errors=None, newline=None)

Open the file pointed to by this path and return a file object, as
the built-in open() function does.

#### owner(, follow_symlinks=True)

Return the login name of the file owner.

#### read_bytes()

Open the file in bytes mode, read it, and close the file.

#### read_text(encoding=None, errors=None, newline=None)

Open the file in text mode, read it, and close the file.

#### readlink()

Return the path to which the symbolic link points.

#### rename(target)

Rename this path to the target path.

The target path may be absolute or relative. Relative paths are
interpreted relative to the current working directory, *not* the
directory of the Path object.

Returns the new Path instance pointing to the target path.

#### replace(target)

Rename this path to the target path, overwriting if that path exists.

The target path may be absolute or relative. Relative paths are
interpreted relative to the current working directory, *not* the
directory of the Path object.

Returns the new Path instance pointing to the target path.

#### resolve(strict=False)

Make the path absolute, resolving all symlinks on the way and also
normalizing it.

#### rglob(pattern, , case_sensitive=None, recurse_symlinks=False)

Recursively yield all existing files (of any kind, including
directories) matching the given relative pattern, anywhere in
this subtree.

#### rmdir()

Remove this directory.  The directory must be empty.

#### samefile(other_path)

Return whether other_path is the same or not as this file
(as returned by os.path.samefile()).

#### stat(, follow_symlinks=True)

Return the result of the stat() system call on this path, like
os.stat() does.

#### symlink_to(target, target_is_directory=False)

Make this path a symlink pointing to the target path.
Note the order of arguments (link, target) is the reverse of os.symlink.

#### touch(mode=438, exist_ok=True)

Create this file with the given access mode, if it doesn’t exist.

#### unlink(missing_ok=False)

Remove this file or link.
If the path is a directory, use rmdir() instead.

#### walk(top_down=True, on_error=None, follow_symlinks=False)

Walk the directory tree from this directory, similar to os.walk().

#### write_bytes(data)

Open the file in bytes mode, write to it, and close the file.

#### write_text(data, encoding=None, errors=None, newline=None)

Open the file in text mode, write to it, and close the file.

### *class* olm.train.callbacks.checkpoint_cb.TrainerCallback

Bases: `object`

Base class for trainer callbacks.

#### on_batch_begin(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), batch_idx: int) → None

Called at the beginning of each batch.

#### on_batch_end(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), batch_idx: int, loss: float) → None

Called at the end of each batch.

#### on_epoch_begin(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), epoch: int) → None

Called at the beginning of each epoch.

#### on_epoch_end(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), epoch: int) → None

Called at the end of each epoch.

#### on_step_begin(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), step: int) → None

Called at the beginning of each optimization step (after gradient accumulation).

#### on_step_end(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), step: int, loss: float) → None

Called at the end of each optimization step.

#### on_train_begin(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer)) → None

Called at the beginning of training.

#### on_train_end(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer)) → None

Called at the end of training.
