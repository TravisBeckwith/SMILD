"""
tests/test_smild_poc.py
"""
import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/claude/smild_repo')

from smild.forward import forward_powder, powder_stick, powder_zeppelin
from smild.twin import degenerate_twin
from smild.smild import smild

B = np.array([0.5, 1.0, 2.0, 3.0])
SIGMA = np.full_like(B, 0.05)

class TestForwardModel:
    def test_stick_at_zero_b(self):
        S = powder_stick(np.array([1e-6]), Da=2.0)
        assert np.isclose(S[0], 1.0, atol=1e-4)

    def test_zeppelin_at_zero_b(self):
        S = powder_zeppelin(np.array([1e-6]), De_par=1.5, De_perp=0.5)
        assert np.isclose(S[0], 1.0, atol=1e-4)

    def test_forward_powder_normalization(self):
        S = forward_powder(np.array([1e-6]), f=0.5, Da=2.0, De_par=1.5, De_perp=0.5)
        assert np.isclose(S[0], 1.0, atol=1e-4)

    def test_signal_monotonically_decreasing(self):
        b = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
        S = forward_powder(b, f=0.5, Da=2.0, De_par=1.5, De_perp=0.5)
        assert np.all(np.diff(S) < 0)

    def test_signal_in_unit_interval(self):
        S = forward_powder(B, f=0.5, Da=2.0, De_par=1.5, De_perp=0.5)
        assert np.all(S > 0) and np.all(S <= 1.0 + 1e-9)

    def test_invalid_f_raises(self):
        with pytest.raises(ValueError):
            forward_powder(B, f=0.0, Da=2.0, De_par=1.5, De_perp=0.5)

    def test_invalid_De_perp_raises(self):
        with pytest.raises(ValueError):
            forward_powder(B, f=0.5, Da=2.0, De_par=0.5, De_perp=0.6)

class TestDegenerateTwin:
    def test_twin_preserves_M1(self):
        f, Da, De_par = 0.4, 2.5, 1.2
        t = degenerate_twin(f, Da, De_par)
        M1_orig = f * Da + (1 - f) * De_par
        M1_twin = f * t.Da_twin + (1 - f) * t.De_par_twin
        assert np.isclose(M1_orig, M1_twin, rtol=1e-10)

    def test_twin_preserves_V(self):
        f, Da, De_par = 0.4, 2.5, 1.2
        t = degenerate_twin(f, Da, De_par)
        V_orig = f * (1 - f) * (Da - De_par) ** 2
        V_twin = f * (1 - f) * (t.Da_twin - t.De_par_twin) ** 2
        assert np.isclose(V_orig, V_twin, rtol=1e-10)

    def test_twin_flips_gap_sign(self):
        f, Da, De_par = 0.5, 2.2, 1.4
        t = degenerate_twin(f, Da, De_par)
        assert (t.Da_twin - t.De_par_twin) < 0

    def test_twin_of_twin_is_original(self):
        f, Da, De_par = 0.45, 2.3, 1.3
        t1 = degenerate_twin(f, Da, De_par)
        t2 = degenerate_twin(f, t1.Da_twin, t1.De_par_twin)
        assert np.isclose(t2.Da_twin, Da, rtol=1e-10)
        assert np.isclose(t2.De_par_twin, De_par, rtol=1e-10)

    def test_zero_gap_twin_is_identical(self):
        D = 1.8
        t = degenerate_twin(0.5, D, D)
        assert np.isclose(t.Da_twin, D, rtol=1e-10)
        assert np.isclose(t.gap, 0.0, atol=1e-12)

class TestSMILD:
    def test_smild_in_unit_interval(self):
        r = smild(B, f=0.5, Da=2.2, De_par=1.5, De_perp=0.6, sigma_S=SIGMA)
        assert 0.0 <= r.smild <= 1.0

    def test_smild_approaches_one_at_zero_gap(self):
        D = 1.8
        r = smild(B, f=0.5, Da=D+1e-6, De_par=D-1e-6, De_perp=0.5, sigma_S=SIGMA)
        assert r.smild > 0.99

    def test_smild_decreases_with_separation(self):
        seps = [0.05, 0.2, 0.5, 1.0, 1.4]
        M1 = 1.9
        vals = [smild(B, f=0.5, Da=M1+0.5*s, De_par=M1-0.5*s,
                      De_perp=0.6, sigma_S=SIGMA).smild for s in seps]
        assert all(a >= b for a, b in zip(vals, vals[1:]))

    def test_smild_decreases_with_snr(self):
        snrs = [8, 15, 30, 60]
        vals = [smild(B, f=0.5, Da=2.25, De_par=1.55, De_perp=0.6,
                      sigma_S=np.full_like(B, 1.0/(snr*np.sqrt(6)))).smild
                for snr in snrs]
        assert all(a >= b for a, b in zip(vals, vals[1:]))

    def test_delta_R_and_smild_consistent(self):
        r = smild(B, f=0.5, Da=2.0, De_par=1.5, De_perp=0.6, sigma_S=SIGMA)
        assert np.isclose(r.smild, np.exp(-0.5 * r.delta_R**2), rtol=1e-9)

    def test_scalar_sigma_broadcast(self):
        r = smild(B, f=0.5, Da=2.0, De_par=1.5, De_perp=0.6, sigma_S=0.05)
        assert 0.0 <= r.smild <= 1.0
