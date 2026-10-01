"""Intensity signal chain of the study, as an importable module.

The chain runs once per audio frame of 50 ms:

    band-pass 700-4500 Hz (two RBJ biquads)
    -> YIN pitch estimate with a confidence gate
    -> median of the last three valid f0 estimates
    -> Hann-windowed Goertzel tone amplitude at that frequency, three probes
       one bin apart, maximum taken, scaled by 2 / L
    -> energy average over the last three frames
    -> force = 100 (dB - dB_min) / (dB_max - dB_min), clipped to 0..100
    -> median of three -> One Euro filter

A whistle that stops for six frames in a row (300 ms) resets the smoothing
state, so that a new whistle does not start from the level of the old one.

The module drives this chain with synthetic signals. Everything is vectorised
over a batch of traces of equal length; the per-frame state (median buffers,
energy window, One Euro filter, reset counter) is stepped frame by frame for
all traces at once.

The application works in 32-bit floats. Here the audio samples and the
band-passed samples are stored as 32-bit floats as well (this also reproduces
the exact zeros that follow a silent input), and the arithmetic is done in
64-bit floats. The tone amplitudes agree with those of the 32-bit
implementation to within 0.001 dB, except in rare frames where the probes miss
the tone by more than a bin and the reading lies far down the skirt of the
window response (more than 15 dB below the tone level).

Only numpy and scipy are needed.
"""
import numpy as np
from scipy import signal as _signal

# ---------------------------------------------------------------------------
# Parameters of the study
# ---------------------------------------------------------------------------
FS = 48000                  # sampling rate (Hz); the rate check gave mu = 1.0000
FRAME = FS // 20            # samples per frame: 2400 (50 ms)
BIN_HZ = FS / FRAME         # Goertzel bin spacing: 20 Hz
BAND_HZ = (700.0, 4500.0)   # band-pass edges
BAND_Q = 0.707
YIN_THRESHOLD = 0.25        # CMND threshold; a frame is valid when confidence >= 0.75
MIN_CONFIDENCE = 0.75
PROBE_STEP_HZ = 20.0        # the three Goertzel probes are f0 - 20, f0, f0 + 20 Hz
ENERGY_FRAMES = 3           # energy window over the last three frames (~150 ms)
MAX_INVALID_FRAMES = 6      # six invalid frames in a row (300 ms) reset the state
ONE_EURO = dict(min_cutoff=2.0, beta=0.002, d_cutoff=1.0)
FILTER_DT = 0.05            # the One Euro filter is stepped once per 50 ms frame

# Probe on the band-passed frame at the median f0 is the chain of the study.
# The other probe choices are counterfactuals used to separate the error sources.
VARIANTS = {
    "task": ("median", "filtered"),        # the chain as used in the intensity tasks
    "true": ("true", "filtered"),          # probe at the true tone frequency
    "raw": ("raw", "filtered"),            # probe at the current YIN estimate (open-loop recordings)
    "true_unfiltered": ("true", "unfiltered"),  # true frequency, no band-pass: scalloping only
    "ideal": (None, None),                 # tone amplitude replaced by a constant
}

_n = np.arange(FRAME)
# Symmetric Hann window (denominator L - 1), as in the Goertzel detector.
HANN = 0.5 * (1.0 - np.cos(2.0 * np.pi * _n / (FRAME - 1)))
HANN_SUM = HANN.sum()
# Reading of a bin-centred tone of unit peak amplitude: (2 / L) * (1/2) * sum(w).
COHERENT_GAIN = HANN_SUM / FRAME


# ---------------------------------------------------------------------------
# Band-pass filter
# ---------------------------------------------------------------------------
def rbj_coefficients(kind, fc, fs=FS, q=BAND_Q):
    """RBJ cookbook biquad (b, a), with the coefficients rounded to 32 bits in
    the same order of operations as the application, so that the frequency
    response is that of the filter the study used."""
    f32 = np.float32
    om = f32(2) * f32(np.pi) * f32(fc) / f32(fs)
    alpha = f32(np.sin(np.float64(om))) / (f32(2) * f32(q))
    cw = f32(np.cos(np.float64(om)))
    if kind == "lowpass":
        b = [(f32(1) - cw) / f32(2), f32(1) - cw, (f32(1) - cw) / f32(2)]
    elif kind == "highpass":
        b = [(f32(1) + cw) / f32(2), -(f32(1) + cw), (f32(1) + cw) / f32(2)]
    else:
        raise ValueError(kind)
    a0 = f32(1) + alpha
    b = np.array([bb / a0 for bb in b], np.float32).astype(np.float64)
    a = np.array([f32(1), (f32(-2) * cw) / a0, (f32(1) - alpha) / a0], np.float32).astype(np.float64)
    return b, a


_HP = rbj_coefficients("highpass", BAND_HZ[0])
_LP = rbj_coefficients("lowpass", BAND_HZ[1])


def bandpass(x):
    """Band-pass a batch of continuous signals (last axis = time).

    The filter state runs on across frame boundaries, so filtering the whole
    signal at once is the same as filtering it frame by frame."""
    y = _signal.lfilter(_HP[0], _HP[1], np.asarray(x, np.float64), axis=-1)
    y = _signal.lfilter(_LP[0], _LP[1], y, axis=-1)
    return y.astype(np.float32)


def bandpass_response_db(f):
    """Steady-state gain of the band-pass filter at frequency f (dB)."""
    f = np.atleast_1d(np.asarray(f, np.float64))
    w = 2.0 * np.pi * f / FS
    _, hh = _signal.freqz(_HP[0], _HP[1], worN=w)
    _, hl = _signal.freqz(_LP[0], _LP[1], worN=w)
    return 20.0 * np.log10(np.abs(hh * hl))


# ---------------------------------------------------------------------------
# Goertzel tone amplitude
# ---------------------------------------------------------------------------
def probe_bins(f0):
    """Bin indices of the three probes for estimated frequency f0.

    Each probe frequency is rounded to its own bin, half to even, as the
    application does. Returns an integer array with a last axis of length 3."""
    f0 = np.asarray(f0, np.float64)
    offs = np.array([-PROBE_STEP_HZ, 0.0, PROBE_STEP_HZ])
    return np.round((f0[..., None] + offs) * FRAME / FS).astype(np.int64)


def frame_spectra(frames):
    """|DFT| of the Hann-windowed frames at every integer bin 0..L/2.

    The Goertzel recursion at integer bin k returns exactly this magnitude, so
    one FFT per frame gives all the probes."""
    return np.abs(np.fft.rfft(np.asarray(frames, np.float64) * HANN, axis=-1))


def _fold_bins(k):
    """Map any bin index onto 0..L/2 (the spectrum of a real frame is symmetric)."""
    k = np.mod(k, FRAME)
    return np.where(k > FRAME // 2, FRAME - k, k)


def tone_amplitude(mag, f0):
    """Tone amplitude of each frame: (2 / L) times the largest of the three probes.

    mag: (..., L/2 + 1) spectra from frame_spectra; f0: (...) probe frequency.
    The factor 2 offsets the coherent gain of the Hann window; it cancels in the
    dB mapping. Frames with f0 <= 0 read 0."""
    k = _fold_bins(probe_bins(f0))
    vals = np.take_along_axis(mag, k, axis=-1).max(axis=-1)
    return np.where(np.asarray(f0) > 0, 2.0 * vals / FRAME, 0.0)


def hann_response(delta_bins):
    """|W(d)| / |W(0)| of the symmetric Hann window, d = offset in bins (exact DTFT)."""
    d = np.atleast_1d(np.asarray(delta_bins, np.float64))
    flat = d.ravel()
    u, inv = np.unique(np.round(flat, 12), return_inverse=True)
    out = np.empty(u.size)
    for s in range(0, u.size, 512):
        blk = u[s:s + 512]
        out[s:s + 512] = np.abs(np.exp(-2j * np.pi * np.outer(blk, _n) / FRAME) @ HANN) / HANN_SUM
    return out[inv].reshape(d.shape)


def scalloping_db(f_true, f_probe=None):
    """Scalloping loss (dB, <= 0) of the three-probe maximum for a pure tone.

    With f_probe = None the probe is centred on the true frequency, which is the
    case whenever the estimate is within 20 Hz of the tone: the nearest bin is
    then one of the three probes and gives the largest reading."""
    f_true = np.atleast_1d(np.asarray(f_true, np.float64))
    if f_probe is None:
        d = f_true / BIN_HZ - np.round(f_true / BIN_HZ)
        return 20.0 * np.log10(hann_response(d))
    k = probe_bins(np.broadcast_to(np.asarray(f_probe, np.float64), f_true.shape))
    r = hann_response(f_true[..., None] / BIN_HZ - k).max(axis=-1)
    return 20.0 * np.log10(r)


def estimator_gain_db(f):
    """Predicted frequency-dependent gain of the intensity reading (dB):
    Hann scalloping at the nearest bin plus the band-pass response."""
    return scalloping_db(f) + bandpass_response_db(f)


# ---------------------------------------------------------------------------
# YIN
# ---------------------------------------------------------------------------
def yin(frames, threshold=YIN_THRESHOLD, chunk=2000):
    """YIN pitch estimate for each frame (last axis = the L samples of a frame).

    Difference function over lags 1..L/2-1 with an L/2-sample window, cumulative
    mean normalisation, the first lag below the threshold followed down to its
    local minimum, and parabolic interpolation. Confidence = 1 - CMND at that
    lag, so every frame that finds a lag has confidence > 0.75.
    Returns (f0, confidence); f0 = -1 and confidence = 0 when no lag is found."""
    frames = np.asarray(frames)
    shape = frames.shape[:-1]
    x_all = frames.reshape(-1, FRAME)
    m = FRAME // 2
    f0 = np.full(x_all.shape[0], -1.0)
    conf = np.zeros(x_all.shape[0])
    tau_idx = np.arange(m)
    for s in range(0, x_all.shape[0], chunk):
        x = x_all[s:s + chunk].astype(np.float64)
        nb = x.shape[0]
        # d(tau) = sum_{j<m} (x_j - x_{j+tau})^2 = e0 + e(tau) - 2 c(tau); j + tau < L,
        # so a circular correlation of length L has no wrap-around.
        a = np.zeros_like(x)
        a[:, :m] = x[:, :m]
        c = np.fft.irfft(np.conj(np.fft.rfft(a, axis=1)) * np.fft.rfft(x, axis=1), FRAME, axis=1)[:, :m]
        cs = np.concatenate([np.zeros((nb, 1)), np.cumsum(x * x, axis=1)], axis=1)
        e0 = cs[:, m]
        et = cs[:, m:2 * m] - cs[:, :m]
        d = np.maximum(e0[:, None] + et - 2.0 * c, 0.0)
        d[:, 0] = 0.0
        run = np.cumsum(d[:, 1:], axis=1)
        cmnd = np.ones((nb, m))
        with np.errstate(invalid="ignore", divide="ignore"):
            cmnd[:, 1:] = np.where(run == 0, 1.0, d[:, 1:] * tau_idx[1:] / np.where(run == 0, 1.0, run))
        below = cmnd < threshold
        below[:, :2] = False
        below[:, m - 1:] = False
        found = below.any(axis=1)
        t0 = np.argmax(below, axis=1)
        stop = np.ones((nb, m), bool)
        stop[:, :-1] = cmnd[:, 1:] >= cmnd[:, :-1]
        stop &= tau_idx[None, :] >= t0[:, None]
        tau = np.argmax(stop, axis=1)
        rows = np.arange(nb)
        inner = (tau > 0) & (tau < m - 1)
        s0 = cmnd[rows, np.clip(tau - 1, 0, m - 1)]
        s1 = cmnd[rows, tau]
        s2 = cmnd[rows, np.clip(tau + 1, 0, m - 1)]
        den = 2.0 * (2.0 * s1 - s2 - s0)
        use = inner & (np.abs(den) >= 1e-6)
        better = tau + np.where(use, (s2 - s0) / np.where(use, den, 1.0), 0.0)
        ok = found & (better > 0)
        f0[s:s + nb] = np.where(ok, FS / np.where(ok, better, 1.0), -1.0)
        conf[s:s + nb] = np.where(ok, 1.0 - s1, 0.0)
    return f0.reshape(shape), conf.reshape(shape)


# ---------------------------------------------------------------------------
# Synthetic sources
# ---------------------------------------------------------------------------
def synthesize(track, amplitude, n_frames, preroll=0, phase0=0.3, gate=None):
    """Tone of constant amplitude that follows a frequency track.

    track: a number (steady tone) or a function f(t) in Hz, t in seconds from the
    start of the analysed frames (the pre-roll has t < 0). The phase advances by
    2*pi*f/FS per sample, with f taken at the sample midpoint, so the frequency
    can change within a frame. gate: optional function g(t) -> 0/1 evaluated at
    each frame centre; a frame with g = 0 is silent (the phase runs on).
    Returns (samples as float32, true frequency at each frame centre; 0 in
    silent frames), both including the pre-roll."""
    total = preroll + n_frames
    n = np.arange(total * FRAME)
    tc = (np.arange(total) * FRAME - preroll * FRAME + (FRAME - 1) / 2.0) / FS
    if np.isscalar(track):
        phase = 2.0 * np.pi * float(track) * n / FS + phase0
        f_centre = np.full(total, float(track))
    else:
        tm = (n - preroll * FRAME + 0.5) / FS
        inc = 2.0 * np.pi * np.asarray(track(tm), np.float64) / FS
        phase = phase0 + np.concatenate([[0.0], np.cumsum(inc[:-1])])
        f_centre = np.asarray(track(tc), np.float64)
    x = (amplitude * np.sin(np.mod(phase, 2.0 * np.pi))).astype(np.float32)
    if gate is not None:
        g = np.asarray(gate(tc)) != 0
        x.reshape(total, FRAME)[~g] = 0.0
        f_centre = np.where(g, f_centre, 0.0)
    return x, f_centre


def interp_track(t_points, f_points):
    """Piecewise-linear frequency track through (t, f) points, held constant
    beyond the first and last point."""
    tp = np.asarray(t_points, np.float64)
    fp = np.asarray(f_points, np.float64)
    return lambda t: np.interp(t, tp, fp)


def nearest_gate(t_points, g_points):
    """Gate that takes, at time t, the 0/1 value of the nearest point (a tie
    goes to the earlier point)."""
    tp = np.asarray(t_points, np.float64)
    gp = np.asarray(g_points)

    def gate(t):
        t = np.asarray(t, np.float64)
        hi = np.clip(np.searchsorted(tp, t, side="right"), 1, len(tp) - 1)
        lo = hi - 1
        pick = np.where(t - tp[lo] <= tp[hi] - t, lo, hi)
        pick = np.where(t <= tp[0], 0, np.where(t >= tp[-1], len(tp) - 1, pick))
        return gp[pick]
    return gate


# ---------------------------------------------------------------------------
# Force mapping and smoothing (vectorised over traces)
# ---------------------------------------------------------------------------
def amplitude_to_force(amp, db_min, db_max):
    """dB mapping of the tone amplitude onto 0..100 force points."""
    amp = np.asarray(amp, np.float64)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(amp, 1e-9))
    force = np.clip(100.0 * (db - db_min) / (db_max - db_min), 0.0, 100.0)
    return np.where(amp > 0, force, 0.0)


class _Median3:
    """Median of the last three values; the first value fills the window."""

    def __init__(self, shape):
        self.buf = np.zeros(shape + (3,))
        self.count = np.zeros(shape, np.int64)

    def step(self, x, mask):
        warm = mask & (self.count < 3)
        run = mask & ~warm
        self.buf[warm] = x[warm][..., None]
        self.count[warm] += 1
        b = self.buf
        b[run] = np.stack([b[run, 1], b[run, 2], x[run]], axis=-1)
        med = np.median(b, axis=-1)
        return np.where(warm, x, med)

    def reset_to(self, value, mask):
        self.buf[mask] = value
        self.count[mask] = 3

    def reset_warmup(self, mask):
        self.count[mask] = 0


class _OneEuro:
    """One Euro filter (Casiez et al., 2012): the cutoff rises with the speed of
    the signal, which gives a steady output at rest and little lag in motion."""

    def __init__(self, shape, min_cutoff, beta, d_cutoff):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = np.zeros(shape)
        self.dx = np.zeros(shape)
        self.init = np.zeros(shape, bool)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def step(self, x, dt, mask):
        first = mask & ~self.init
        run = mask & self.init
        dx = (x - self.x) / dt
        a_d = self._alpha(self.dc, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx
        a = self._alpha(self.mc + self.beta * np.abs(dx_hat), dt)
        x_hat = a * x + (1.0 - a) * self.x
        self.x = np.where(first, x, np.where(run, x_hat, self.x))
        self.dx = np.where(first, 0.0, np.where(run, dx_hat, self.dx))
        self.init |= mask
        return self.x

    def reset_to(self, value, mask):
        self.init[mask] = True
        self.x[mask] = value
        self.dx[mask] = 0.0


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------
def run_chain(x, bounds, true_f0=None, ideal_tone=None, variants=("task",), spectrum_chunk=4000):
    """Run the per-frame chain over a batch of signals.

    x: (T, n_frames * FRAME) samples. bounds: (T, M, 2) calibrated bounds
    [A_min, A_max] (linear) of M force mappings evaluated side by side.
    true_f0: (T, n_frames) true frequency at each frame centre (needed by the
    'true' variants). ideal_tone: (T,) constant tone amplitude of the 'ideal'
    variant. All variants share the band-pass filter, YIN and the validity
    gate; each has its own median, energy window and smoothing state.

    Returns a dict with 'valid', 'confidence', 'f0_raw' (T, F) and, for each
    variant v, 'v_probe', 'v_tone', 'v_amp' (T, F) and 'v_force' (T, F, M).
    As in the application, the outputs hold their last value through invalid
    frames and drop to 0 when six invalid frames in a row reset the state."""
    x = np.asarray(x, np.float32)
    T = x.shape[0]
    F = x.shape[1] // FRAME
    bounds = np.asarray(bounds, np.float64).reshape(T, -1, 2)
    M = bounds.shape[1]
    db_min = 20.0 * np.log10(np.maximum(bounds[..., 0], 1e-9))
    db_max = 20.0 * np.log10(np.maximum(bounds[..., 1], 1e-9))

    filtered = bandpass(x).reshape(T, F, FRAME)
    f_raw, conf = yin(filtered)
    valid = conf >= MIN_CONFIDENCE

    # Pass 1: validity counter, resets and the probe frequency of each variant.
    reset = np.zeros((T, F), bool)
    invalid_run = np.zeros(T, np.int64)
    med_f0 = _Median3((T,))
    probes = {}
    for v in variants:
        src = VARIANTS[v][0]
        if src == "true" and true_f0 is None:
            raise ValueError("variant %s needs true_f0" % v)
        if src is not None:
            probes[v] = np.zeros((T, F))
    for k in range(F):
        ok = valid[:, k]
        invalid_run = np.where(ok, 0, invalid_run + 1)
        rs = ~ok & (invalid_run >= MAX_INVALID_FRAMES)
        reset[:, k] = rs
        med_f0.reset_warmup(rs)
        mf = med_f0.step(f_raw[:, k], ok)
        for v in probes:
            src = VARIANTS[v][0]
            val = {"median": mf, "raw": f_raw[:, k], "true": None if true_f0 is None else true_f0[:, k]}[src]
            probes[v][:, k] = np.where(ok, val, 0.0)

    # Tone amplitudes of the valid frames, from one spectrum per frame.
    tones = {v: np.zeros((T, F)) for v in variants}
    need = {VARIANTS[v][1] for v in variants if VARIANTS[v][1] is not None}
    flat_valid = np.flatnonzero(valid.ravel())
    sources = {"filtered": filtered.reshape(T * F, FRAME)}
    if "unfiltered" in need:
        sources["unfiltered"] = x.reshape(T * F, FRAME)
    for s in range(0, flat_valid.size, spectrum_chunk):
        idx = flat_valid[s:s + spectrum_chunk]
        mags = {b: frame_spectra(sources[b][idx]) for b in need}
        for v in probes:
            tones[v].reshape(-1)[idx] = tone_amplitude(mags[VARIANTS[v][1]], probes[v].reshape(-1)[idx])
    if "ideal" in variants:
        if ideal_tone is None:
            raise ValueError("variant ideal needs ideal_tone")
        tones["ideal"] = np.where(valid, np.asarray(ideal_tone, np.float64)[:, None], 0.0)

    # Pass 2: energy window, force mapping, median of three, One Euro.
    out = {"valid": valid, "confidence": conf, "f0_raw": f_raw}
    for v in variants:
        ms = np.zeros((T, ENERGY_FRAMES))
        pos = np.zeros(T, np.int64)
        cnt = np.zeros(T, np.int64)
        med = _Median3((T, M))
        euro = _OneEuro((T, M), **ONE_EURO)
        amp_hold = np.zeros(T)
        probe_hold = np.zeros(T)
        tone_hold = np.zeros(T)
        force_hold = np.zeros((T, M))
        amps = np.zeros((T, F))
        prb = np.zeros((T, F))
        tns = np.zeros((T, F))
        frc = np.zeros((T, F, M))
        rows = np.arange(T)
        slots = np.arange(ENERGY_FRAMES)
        for k in range(F):
            ok = valid[:, k]
            rs = reset[:, k]
            if rs.any():
                force_hold[rs] = 0.0
                med.reset_to(0.0, np.repeat(rs[:, None], M, axis=1))
                euro.reset_to(0.0, np.repeat(rs[:, None], M, axis=1))
                pos[rs] = 0
                cnt[rs] = 0
                amp_hold[rs] = probe_hold[rs] = tone_hold[rs] = 0.0
            if ok.any():
                tone = tones[v][:, k]
                ms[rows[ok], pos[ok]] = tone[ok] ** 2
                pos[ok] = (pos[ok] + 1) % ENERGY_FRAMES
                cnt[ok] = np.minimum(cnt[ok] + 1, ENERGY_FRAMES)
                used = slots[None, :] < cnt[:, None]
                amp = np.sqrt((ms * used).sum(axis=1) / np.maximum(cnt, 1))
                amp_hold = np.where(ok, amp, amp_hold)
                tone_hold = np.where(ok, tone, tone_hold)
                if v in probes:
                    pv = probes[v][:, k]
                else:
                    pv = true_f0[:, k] if true_f0 is not None else np.zeros(T)
                probe_hold = np.where(ok, pv, probe_hold)
                raw_force = amplitude_to_force(amp_hold[:, None], db_min, db_max)
                okm = np.repeat(ok[:, None], M, axis=1)
                mf = med.step(raw_force, okm)
                ef = euro.step(mf, FILTER_DT, okm)
                force_hold = np.where(okm, ef, force_hold)
            amps[:, k] = amp_hold
            prb[:, k] = probe_hold
            tns[:, k] = tone_hold
            frc[:, k] = force_hold
        out[v + "_amp"] = amps
        out[v + "_probe"] = prb
        out[v + "_tone"] = tns
        out[v + "_force"] = frc
    return out
