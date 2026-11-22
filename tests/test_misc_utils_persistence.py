"""Tests for ``mcrotk_utils.misc_utils``' persistence: ``write_atomically``."""

import os
import pickle
import stat

import pytest

from mcrotk_utils.misc_utils import _persistence, write_atomically


def test_write_atomically_writes_bytes_and_text(tmp_path):
    """Bytes are written as they are; text as UTF-8."""
    write_atomically(tmp_path / "data.bin", b"\x00\x01")
    assert (tmp_path / "data.bin").read_bytes() == b"\x00\x01"
    write_atomically(str(tmp_path / "data.txt"), "naïve ✓")
    assert (tmp_path / "data.txt").read_bytes() == "naïve ✓".encode()


def test_write_atomically_replaces_and_leaves_nothing_else(tmp_path):
    """Writing again replaces the file, and no temporary files remain."""
    path = tmp_path / "results.pkl"
    write_atomically(path, pickle.dumps([1, 2]))
    write_atomically(path, pickle.dumps([3]))
    assert pickle.loads(path.read_bytes()) == [3]
    assert [p.name for p in tmp_path.iterdir()] == ["results.pkl"]


def test_write_atomically_permissions(tmp_path):
    """A new file gets the same permissions as one made by open(); a replaced file keeps its permissions."""
    reference = tmp_path / "reference"
    reference.write_bytes(b"")
    new = tmp_path / "new"
    write_atomically(new, b"x")
    assert stat.S_IMODE(new.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)
    os.chmod(new, 0o640)
    write_atomically(new, b"y")
    assert stat.S_IMODE(new.stat().st_mode) == 0o640


@pytest.mark.parametrize("failure", [OSError("disk full"), KeyboardInterrupt()])
def test_interrupted_write_keeps_the_previous_file(tmp_path, monkeypatch, failure):
    """If writing fails or is interrupted, the previous file is untouched and the temporary file is removed."""
    path = tmp_path / "results.pkl"
    write_atomically(path, b"previous")
    monkeypatch.setattr(_persistence, "_write_and_sync", _raiser(failure))
    with pytest.raises(type(failure)):
        write_atomically(path, b"new")
    assert path.read_bytes() == b"previous"
    assert [p.name for p in tmp_path.iterdir()] == ["results.pkl"]


def test_failed_rename_leaves_no_file_at_a_new_path(tmp_path, monkeypatch):
    """If the rename fails for a new path, there's no file there afterwards, and no temporary file."""
    monkeypatch.setattr(_persistence.os, "replace", _raiser(OSError("rename failed")))
    with pytest.raises(OSError, match="rename failed"):
        write_atomically(tmp_path / "new.pkl", b"data")
    assert list(tmp_path.iterdir()) == []


def test_write_atomically_syncs_before_renaming(tmp_path, monkeypatch):
    """The data is flushed to storage (os.fsync) before the rename."""
    calls = []
    real_fsync, real_replace = os.fsync, os.replace

    def recording_fsync(descriptor):
        """Record the call, then sync."""
        calls.append("fsync")
        real_fsync(descriptor)

    def recording_replace(source, destination):
        """Record the call, then rename."""
        calls.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(_persistence.os, "fsync", recording_fsync)
    monkeypatch.setattr(_persistence.os, "replace", recording_replace)
    write_atomically(tmp_path / "f", b"x")
    assert calls[:2] == ["fsync", "replace"]


def test_write_atomically_replaces_a_symbolic_link(tmp_path):
    """A symbolic link at the path is replaced by the file; the link's target is left alone."""
    linked = tmp_path / "linked"
    linked.write_bytes(b"original")
    link = tmp_path / "link"
    link.symlink_to(linked)
    write_atomically(link, b"new")
    assert not link.is_symlink()
    assert link.read_bytes() == b"new"
    assert linked.read_bytes() == b"original"


def test_write_atomically_needs_an_existing_folder(tmp_path):
    """A missing folder raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        write_atomically(tmp_path / "missing" / "f", b"x")


@pytest.mark.parametrize("data", [bytearray(b"x"), 5, None])
def test_write_atomically_rejects_other_data(tmp_path, data):
    """Data other than bytes or str raises TypeError, and nothing is written."""
    with pytest.raises(TypeError, match="data must be bytes or str"):
        write_atomically(tmp_path / "f", data)
    assert list(tmp_path.iterdir()) == []


def _raiser(exception):
    """Return a function that raises ``exception`` whatever it's called with."""

    def raise_it(*args, **kwargs):
        """Raise the exception."""
        raise exception

    return raise_it
