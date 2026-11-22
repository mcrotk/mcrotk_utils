"""Tests for ``mcrotk_utils.stats_utils.NormalParams``."""

import dataclasses

import numpy as np
import pytest

from mcrotk_utils.stats_utils import (
    HypothesisType,
    NormalParams,
)

UPPER, LOWER, TWO_SIDED = HypothesisType.ONE_SIDED_UPPER, HypothesisType.ONE_SIDED_LOWER, HypothesisType.TWO_SIDED


# --- NormalParams ------------------------------------------------------------------------------------------------


def test_normal_params_defaults_and_fields():
    """The defaults are mean 0 and SD 1; the fields keep the values given, positionally or by name."""
    assert (NormalParams().mean, NormalParams().sd) == (0.0, 1.0)
    assert NormalParams(-2.5, 0.1) == NormalParams(mean=-2.5, sd=0.1)
    assert (NormalParams(sd=3).mean, NormalParams(sd=3).sd) == (0.0, 3)


def test_normal_params_are_immutable_and_hashable():
    """Assigning to a field raises (frozen), and equal parameters hash equally."""
    params = NormalParams(1.0, 2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.sd = 3.0  # type: ignore[misc]
    assert hash(params) == hash(NormalParams(1.0, 2.0))
    assert len({params, NormalParams(1.0, 2.0), NormalParams()}) == 2


@pytest.mark.parametrize(
    ("mean", "sd", "message"),
    [(0.0, -0.1, "sd must be non-negative"), (np.nan, 1.0, "must be finite"), (0.0, np.inf, "must be finite"),
     (-np.inf, 1.0, "must be finite"), (0.0, np.nan, "must be finite")],
)  # fmt: skip
def test_normal_params_reject_bad_values(mean, sd, message):
    """A non-finite mean or SD, or a negative SD, raise ValueError."""
    with pytest.raises(ValueError, match=message):
        NormalParams(mean, sd)


def test_normal_params_sd_zero_gives_no_randomness():
    """An SD of 0 is allowed: every normal draw is the mean, and every lognormal one exp(mean)."""
    params = NormalParams(1.5, 0.0)
    assert np.all(params.generate_normal(5, rng=1) == 1.5)
    assert np.allclose(params.generate_lognormal(5, rng=1), np.exp(1.5), rtol=1e-15)


@pytest.mark.parametrize("method", ["normal", "lognormal"])
@pytest.mark.parametrize("size", [None, 0, 1, 7, (2, 3)])
def test_normal_params_generate_numpys_draws(method, size):
    """The draws are numpy's Generator.normal/lognormal with these parameters: a float for size=None, otherwise an
    array of that size."""
    params = NormalParams(0.5, 0.8)
    result = getattr(params, f"generate_{method}")(size, rng=11)
    expected = getattr(np.random.default_rng(11), method)(0.5, 0.8, size)
    if size is None:
        assert type(result) is float
        assert result == expected
    else:
        assert isinstance(result, np.ndarray) and result.dtype == np.float64
        assert result.shape == np.shape(expected)
        assert np.array_equal(result, expected)


def test_normal_params_lognormal_is_exp_of_normal():
    """With the same rng, the lognormal draws are exp of the normal draws."""
    params = NormalParams(-2.5, 0.4)
    assert np.allclose(params.generate_lognormal(100, rng=12), np.exp(params.generate_normal(100, rng=12)), rtol=1e-15)


def test_normal_params_generators_continue_their_stream():
    """A Generator passed as rng is used, and advanced, rather than re-seeded."""
    generator = np.random.default_rng(13)
    params = NormalParams()
    first, second = params.generate_normal(3, rng=generator), params.generate_normal(3, rng=generator)
    assert not np.array_equal(first, second)
    assert np.array_equal(np.concatenate([first, second]), np.random.default_rng(13).normal(0, 1, 6))
    assert params.generate_normal(rng=14) == params.generate_normal(rng=14)


def test_normal_params_rng_is_keyword_only():
    """rng must be given by keyword."""
    with pytest.raises(TypeError):
        NormalParams().generate_normal(3, 5)  # type: ignore[call-overload]
