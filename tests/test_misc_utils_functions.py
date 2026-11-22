"""Tests for ``mcrotk_utils.misc_utils``' ``identity``."""

import pickle

import pytest

from mcrotk_utils.misc_utils import identity


def test_identity_returns_its_argument():
    """The argument comes back as the same object."""
    value = [1, 2]
    assert identity(value) is value
    assert identity(None) is None


def test_identity_can_be_pickled():
    """identity pickles (unlike a lambda), so it can be sent to worker processes."""
    assert pickle.loads(pickle.dumps(identity)) is identity


def test_identity_argument_is_positional_only():
    """The argument can't be passed by name."""
    with pytest.raises(TypeError):
        identity(value=1)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
