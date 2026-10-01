# Column descriptions

Data accompanying *Whistling as a Proportional Switch: Design, Instrumentation,
and Empirical Comparison of Pitch- and Intensity-Based Continuous Control*
(IEEE Access).

## Files

| Path | Contents |
|---|---|
| `participant tests/pa<N>_<channel>_<task>_data.csv` | Main study (N = 1–24), one row per trial |
| `participant tests/pa<N>_<channel>_<task>_raw_data.csv` | Main study, one row per update tick |
| `participant tests/per-participant-summary_data.csv` | One row per participant × channel × task |
| `participant tests/calibration-log_data.csv` | Calibration attempts and accepted bounds |
| `participant tests/openloop-log_data.csv` | Open-loop production, one row per segment |
| `participant tests/openloop-raw_data.csv` | Open-loop production, one row per frame |
| `participant tests/session-log_data.csv` | Capture device and ambient-level check |
| `participant tests/subjective-measures.xlsx` | NASA-TLX, SUS and preference |
| `pilot tests/` | The same per-trial, raw, calibration, open-loop and session files for the three pilot participants (`<N>_<channel>_<task>_…`, N = 1–3), and `subjective-measures-pilot.xlsx` |
| `questionnaires_tr.md` | Turkish wording of the questionnaires as administered |
| `scripts/` | Python scripts that compute the results of the paper; see `scripts/README.md` |
| `LICENSE.md` | Licenses: data CC BY 4.0, scripts MIT |

`<channel>` is `pitch` or `intensity`; `<task>` is `maintenance` or `tracking`.

## Conventions

- **Force** is on a 0–100 point scale; 0 and 100 are the participant's own
  calibrated bounds.
- **Tolerance band:** ±15 points around the target.
- **Trial length:** 6000 ms. **Scoring window:** 1000–6000 ms. All summary
  metrics use the scoring window; raw files contain the whole trial from 0 ms.
- **Update tick:** ≈62.7 ms on average (audio is processed in 50 ms frames).
  Time-based metrics use the actual interval between ticks.
- `TIB` and `Success` are fractions (0–1); the paper reports percentages.
- **Block sequence:** participants `pa1`–`pa12` followed sequence A (pitch
  maintenance, intensity maintenance, pitch tracking, intensity tracking) and
  `pa13`–`pa24` sequence B, the reverse order.
- Period decimal separator, comma-separated, UTF-8. An empty cell means the
  value is not defined for that row.

## Maintenance — `*_maintenance_data.csv` (one row per trial)

| Column | Meaning |
|---|---|
| `Trial` | Trial number in the block, 1–20. |
| `Target` | Target level: 20, 40, 60 or 80 points. |
| `RMSE` | Root-mean-square deviation from the target (points). |
| `SD` | Standard deviation of the produced force (points). |
| `TIB` | Time in band: fraction of the scoring window spent within ±15 points of the target. |
| `Success` | 1 if the force stayed in band for at least 3 s without interruption, otherwise 0. |
| `CompletionTime` | Seconds from the start of the scoring window until the 3 s criterion was met. Empty if it was not met. |
| `Bias` | Mean produced force minus target (points). |
| `LongestStreak` | Longest uninterrupted time in band (s). |
| `TimeToFirstEntry` | Seconds from the start of the scoring window to the first entry into the band. |
| `NSamples` | Number of ticks in the scoring window. |

For a constant target, `RMSE² = SD² + Bias²`.

## Maintenance — `*_maintenance_raw_data.csv` (one row per tick)

| Column | Meaning |
|---|---|
| `Trial`, `Target` | As in the summary file. |
| `SampleIndex` | Tick number within the trial, from 0. |
| `TimeMs` | Time since trial onset (ms). |
| `Force` | Produced force (points). |
| `FrequencyHz` | Estimated fundamental frequency f0. In intensity files it is 0 when no valid estimate was available, and otherwise the median of the last three valid estimates, i.e. the frequency at which `Amplitude` was measured. In pitch files it keeps the last valid estimate while the output is reset; `Force` is 0 on those ticks. |
| `Amplitude` | Intensity files only. Narrow-band tone amplitude at `FrequencyHz`, energy-averaged over the last three frames (linear units); 0 when no valid estimate was available. This is the value that is mapped to force. |

## Tracking — `*_tracking_data.csv` (one row per trial)

| Column | Meaning |
|---|---|
| `Trial` | Trial number in the block, 1–15. |
| `Scenario` | Rate index, 1–5. |
| `SpeedPercent` | Target frequency × 100: 20, 35, 50, 65 or 80 (0.20–0.80 Hz). |
| `RMSE` | Root-mean-square deviation from the instantaneous target (points). |
| `PearsonR` | Correlation between produced force and target at zero lag. |
| `BestLagR` | Highest correlation over shifts of up to ±10 ticks; a shift is used only if the overlap covers at least half of the window. |
| `BestLagMs` | Shift at which `BestLagR` occurs, converted to milliseconds with the trial's mean tick interval. Positive values mean the produced force followed the target. |
| `Gain` | Amplitude of the produced force at the target frequency divided by the target amplitude (30 points). Obtained by a least-squares fit of `a + b·cos(2πft) + c·sin(2πft)` to the produced force over the scoring window: `Gain = √(b² + c²) / 30`. Values below 1 mean the target extremes were not reached. |
| `Bias` | Mean produced force minus mean target (points). |
| `TIB` | Fraction of the scoring window spent within ±15 points of the moving target. |
| `NSamples` | Number of ticks in the scoring window. |

Target: `T(t) = 50 − 30·cos(2π·f·t)` points, with `f = SpeedPercent / 100` Hz and
`t` in seconds from trial onset. The target moves between 20 and 80 and starts at 20.

## Tracking — `*_tracking_raw_data.csv` (one row per tick)

| Column | Meaning |
|---|---|
| `Trial`, `Scenario`, `SpeedPercent` | As in the summary file. |
| `SampleIndex`, `TimeMs` | As in the maintenance raw files. |
| `Actual` | Produced force (points). |
| `Target` | Instantaneous target (points). |
| `FrequencyHz`, `Amplitude` | As in the maintenance raw files. |

## `per-participant-summary_data.csv`

One row per `Participant` × `Channel` × `Task`, holding the mean of that block's
trial values. `CompletionTime` is averaged over successful trials only. Columns
that do not apply to a task are empty.

## `calibration-log_data.csv` (one row per calibration attempt, including rejected ones)

| Column | Meaning |
|---|---|
| `Participant` | Participant code. |
| `Type` | `MinPitch`, `MaxPitch`, `MinIntensity` or `MaxIntensity`. |
| `Accepted` | 1 if the attempt passed the acceptance rules, otherwise 0. |
| `MeasuredValue` | Result of the attempt: MIDI note number (pitch) or linear amplitude (intensity). |
| `Trial1`–`Trial3` | The three per-recording values that were averaged. |
| `Frames1`–`Frames3` | Number of valid frames in each recording. |
| `MinMidi`, `MaxMidi` | Pitch bounds in effect after the attempt (MIDI note numbers). |
| `PitchSpanSemitones` | `MaxMidi − MinMidi`. |
| `MinAmplitude`, `MaxAmplitude` | Intensity bounds in effect after the attempt (linear amplitude). |
| `IntensityRangeDb` | `20·log10(MaxAmplitude / MinAmplitude)`. |
| `Device` | Capture device reported by the operating system. |
| `SampleRateMultiplier` | Sampling-rate correction factor μ from the reference-tone check. |

## `openloop-log_data.csv` (three 6 s segments per participant, without display or target)

| Column | Meaning |
|---|---|
| `Participant`, `Segment` | Participant code and segment number, 1–3. |
| `Frames` | Number of valid frames in the segment. |
| `MedianF0Hz` | Median f0 (Hz). |
| `PitchSdSemitones` | Standard deviation of f0 (semitones). |
| `MeanAmpDbFS`, `AmpSdDb` | Mean and standard deviation of tone amplitude (dB relative to full scale). |
| `PitchSdPoints`, `AmpSdPoints` | The same two standard deviations in force points, using the participant's calibrated bounds. These are the values used for the open-loop comparison in the paper. |
| `PitchSpanSemitones`, `IntensityRangeDb` | Calibrated ranges, repeated for convenience. |
| `Device`, `SampleRateMultiplier` | As in the calibration log. |

## `openloop-raw_data.csv` (one row per frame)

| Column | Meaning |
|---|---|
| `Participant`, `Segment` | As in the open-loop log. |
| `SampleIndex`, `TimeMs` | Frame number and time within the segment (ms). |
| `FrequencyHz` | Estimated f0 (Hz). |
| `Amplitude` | Narrow-band tone amplitude at the estimated f0 (linear units). |

## `session-log_data.csv` (one row per session)

| Column | Meaning |
|---|---|
| `Participant` | Participant code. |
| `Device`, `SampleRateMultiplier` | As in the calibration log. |
| `Frames` | Frames used by the ambient-level check. |
| `BroadbandMeanDbFS`, `BroadbandPeakDbFS` | Ambient level at the microphone before band-pass filtering (dBFS). |
| `BandpassedMeanDbFS`, `BandpassedPeakDbFS` | The same after the 700–4500 Hz band-pass filter. |
| `RoomDbaMin`, `RoomDbaTyp`, `RoomDbaMax` | Room level in dB(A), measured with a hand-held sound level meter. |

## `subjective-measures.xlsx`

| Sheet | Contents |
|---|---|
| `NASA-TLX raw` | One row per participant × block: the six subscales and the raw TLX score. `PE as reported` is on the administered scale (0 = unsuccessful, 100 = very successful); `PE as load` is `100 − x`, the value that enters the raw TLX score. |
| `SUS` | One row per participant × channel: the ten item responses (1–5) and the SUS score (0–100). |
| `Preference` | One row per participant: the forced-choice answer. |
| `Summary` | Group means. |

Participants are coded `P01`–`P24` in the workbook and `pa1`–`pa24` in the CSV
files, with the same numbering. The pilot participants (`1`–`3`) did not take
part in the main study; `subjective-measures-pilot.xlsx` has the same sheets.
