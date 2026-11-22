"""Saving data to files safely: writes that never leave a partial file at the path (private).

Public through ``mcrotk_utils.misc_utils``. Uses only the standard library.
"""

import os
import secrets
from pathlib import Path


def write_atomically(path: str | os.PathLike[str], data: bytes | str) -> None:
    """Write ``data`` to the file at ``path`` so that the path always holds a complete version of the file.

    A plain write empties the file first and then fills it, so an interruption (Ctrl-C, a crash, a full disk)
    leaves a truncated file, and the previous version is lost. Here the data goes to a temporary file in the same
    folder, which is then renamed onto ``path`` (a rename within a folder is atomic on macOS, Linux and Windows).
    So the path holds either the previous file or the new one, never a partial one; an interrupted write to a new
    path leaves no file. That makes "the file exists" a reliable sign that the write finished, e.g. for saving
    results during a long run, and skipping finished work when resuming.

    E.g. ``write_atomically(path, pickle.dumps(results))`` or ``write_atomically(path, json.dumps(data))``.

    **Arguments.**

    - ``path``: the file to write. Its folder must exist. A symbolic link at ``path`` is replaced by the new file,
      not followed.
    - ``data``: the contents: ``bytes``, or a ``str``, written as UTF-8.

    **Result.**

    - ``None``. The file is complete at ``path``, and has been flushed to the operating system's storage
      (``os.fsync``) before the rename. A new file gets the default permissions for new files (as ``open`` would
      give it); a replaced file keeps its permissions. Other metadata (owner, extended attributes) isn't kept.

    **Errors.**

    - ``OSError`` (e.g. ``FileNotFoundError`` if the folder doesn't exist): the temporary file is removed and
      ``path`` is left as it was.
    - ``TypeError``: ``data`` is neither ``bytes`` nor ``str``.
    """
    target = Path(path)
    contents = _as_bytes(data)
    temporary = _create_temporary_file(target)
    try:
        _write_and_sync(temporary, contents)
        _copy_permissions(target, temporary)
        os.replace(temporary, target)
    except BaseException:
        # whatever went wrong (including KeyboardInterrupt), don't leave the temporary file behind
        temporary.unlink(missing_ok=True)
        raise
    _sync_folder(target.parent)


def _as_bytes(data: bytes | str) -> bytes:
    """Return ``data`` as bytes (a ``str`` encoded as UTF-8), raising ``TypeError`` for anything else."""
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, bytes):
        return data
    raise TypeError(f"data must be bytes or str, not {type(data).__name__}.")


def _create_temporary_file(target: Path) -> Path:
    """Create an empty, uniquely named file next to ``target`` and return its path.

    Created with mode 0o666, which the process's umask reduces, so that the file gets the same default
    permissions as one created by ``open`` (``tempfile.mkstemp`` would make it readable by its owner only).
    """
    while True:
        temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:  # pragma: no cover (a clash of 64 random bits)
            continue
        os.close(descriptor)
        return temporary


def _write_and_sync(file_path: Path, contents: bytes) -> None:
    """Write ``contents`` to ``file_path`` and flush it to storage before returning."""
    with open(file_path, "wb") as file:
        file.write(contents)
        file.flush()
        os.fsync(file.fileno())


def _copy_permissions(source: Path, destination: Path) -> None:
    """Give ``destination`` the permissions of ``source``, if ``source`` exists (so that replacing keeps them)."""
    try:
        mode = source.stat().st_mode
    except FileNotFoundError:
        return
    os.chmod(destination, mode & 0o7777)


def _sync_folder(folder: Path) -> None:
    """Flush the folder's entry for the renamed file to storage, where the system allows it (not on Windows)."""
    if os.name != "posix":  # pragma: no cover (Windows can't open folders)
        return
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover (some file systems don't support syncing folders; the file is written anyway)
        pass
    finally:
        os.close(descriptor)
