from av import AudioFrame
import numpy as np
from librosa.effects import pitch_shift
from aiortc import MediaStreamTrack


class AudioTransformTrack(MediaStreamTrack):
    kind = 'audio'

    def __init__(self, track, audio_effect):
        super().__init__()
        self.track = track
        self.audio_effect = audio_effect

    async def recv(self):
        frame = await self.track.recv()
        return await apply_audio_effects(audio_effect=self.audio_effect, frame=frame)


async def apply_audio_effects(frame: AudioFrame, audio_effect='normal'):
    if audio_effect == 'normal':
        return frame

    elif audio_effect == 'robot_voice':
        robot_changer = RobotVoice(frame=frame)
        return await robot_changer.process()

    elif audio_effect == 'alien_voice':
        alien_changer = AlienVoice(frame=frame)
        return await alien_changer.process()

    elif audio_effect == 'chipmunk':
        pitch_changer = AudioPitch2(frame)
        return await pitch_changer.process()


class AlienVoice:
    def __init__(self, frame: AudioFrame, mod_freq: int = 500):
        self.frame = frame
        self.raw_samples = self.frame.to_ndarray()
        self.audio = self.raw_samples[0, :]  # From 2D: (2048, 1) to 1D: (2048, )
        self.mod_freq = mod_freq

    async def process(self):
        ''' Formula for The "Alien Voice":
        The cheapest trick in the book to alter a person's voice is to use
        standard sinusoidal modulation to shift the voice spectrum up or down:
        y[n]=x[n]cos(ω0n)

        x (audio): ndarray dtype64
        mod_freq: int, modulation frequency
        '''
        x = self.audio.astype(np.float64)

        w = (float(self.mod_freq) / self.frame.sample_rate) * 2 * np.pi  # normalized modulation frequency
        alien_voice = 2 * np.multiply(x, np.cos(w * np.arange(0, len(x))))
        alien_voice = alien_voice.astype(np.int16)

        new_samples = alien_voice.reshape(self.raw_samples.shape)  # From 1D to 2D

        new_frame = AudioFrame.from_ndarray(new_samples, layout=self.frame.layout.name)
        new_frame.sample_rate = self.frame.sample_rate
        new_frame.pts = self.frame.pts
        new_frame.time_base = self.frame.time_base
        return new_frame


class AudioPitch2:
    def __init__(self, frame: AudioFrame):
        self.frame = frame
        self.raw_samples = self.frame.to_ndarray()
        self.audio = self.raw_samples[0, :]  # From 2D: (2048, 1) to 1D: (2048, )
        self.octaves = {'chipmunk_9': 9}

    async def process(self):
        # Pitch shifting?  Let's gear-shift by a major third (4 semitones)
        y = self.audio.astype(np.float32)
        chipmunk = pitch_shift(y, self.frame.sample_rate, self.octaves['chipmunk_9'])
        chipmunk = chipmunk.astype(np.int16)

        new_samples = chipmunk.reshape(self.raw_samples.shape)  # From 1D to 2D

        new_frame = AudioFrame.from_ndarray(new_samples, layout=self.frame.layout.name)
        new_frame.sample_rate = self.frame.sample_rate
        new_frame.pts = self.frame.pts
        new_frame.time_base = self.frame.time_base
        return new_frame


class RobotVoice:
    def __init__(
            self, frame: AudioFrame, lookup_samples: int = 1024,
            mod_f: int = 50, vb: float = 0.2, vl: float = 0.4, h: int = 4
    ):
        self.frame = frame
        self.raw_samples = self.frame.to_ndarray()
        self.audio = self.raw_samples[0, :]  # From 2D (2048, 1) -> 1 Dimensional (2048, )

        # CONSTANTS
        self.vb = vb  # Diode constants (must be below 1; paper uses 0.2 and 0.4)
        self.vl = vl
        self.h = h  # Controls distortion
        self.lookup_samples = lookup_samples  # Controls N samples in lookup table; probably leave this alone
        self.mod_f = mod_f  # Frequency (in Hz) of modulating frequency

    async def process(self):
        # get max value to scale to original volume at the end
        scaler = np.max(np.abs(self.audio))
        if scaler == 0:
            scaler = 1

        # Normalize to floats in range -1.0 < data < 1.0
        data = self.audio.astype(float) / scaler

        # Length of array (number of samples)
        n_samples = data.shape[0]

        # Create the lookup table for simulating the diode.
        d_lookup = await diode_lookup(self.lookup_samples, self.vb, self.vl, self.h)

        diode = WaveShaper(d_lookup)

        # Simulate sine wave of frequency mod_f (in Hz)
        tone = np.arange(n_samples)

        tone = np.sin(2 * np.pi * tone * self.mod_f / self.frame.rate)

        # Gain tone by 1/2
        tone = tone * 0.5

        # Junctions here
        tone2 = tone.copy()  # to top path

        data2 = data.copy()  # to bottom path

        # Invert tone, sum paths
        tone = -tone + data2  # bottom path

        data = data + tone2  # top path

        # top
        data = diode.transform(data) + diode.transform(-data)

        # bottom
        tone = diode.transform(tone) + diode.transform(-tone)

        result = data - tone

        # scale to +-1.0
        result /= np.max(np.abs(result))

        # now scale to max value of input file.
        result *= scaler

        result = result.astype(np.int16)

        new_samples = result.reshape(self.raw_samples.shape)

        new_frame = AudioFrame.from_ndarray(new_samples, layout=self.frame.layout.name)
        new_frame.sample_rate = self.frame.sample_rate
        new_frame.pts = self.frame.pts
        new_frame.time_base = self.frame.time_base
        return new_frame


class WaveShaper:
    def __init__(self, curve):
        self.curve = curve
        self.n_bins = self.curve.shape[0]

    def transform(self, samples):
        # normalize to 0 < samples < 2
        max_val = np.max(np.abs(samples))
        if max_val >= 1.0:
            result = samples / np.max(np.abs(samples)) + 1.0
        else:
            result = samples + 1.0

        result = result * (self.n_bins - 1) / 2

        return self.curve[result.astype(np.int16)]


async def diode_lookup(n_samples, vb, vl, h):
    result = np.zeros((n_samples,))
    for i in range(0, n_samples):
        v = float(i - float(n_samples) / 2) / (n_samples / 2)
        v = abs(v)
        if v < vb:
            result[i] = 0
        elif vb < v <= vl:
            result[i] = h * ((v - vb) ** 2) / (2 * vl - 2 * vb)
        else:
            result[i] = h * v - h * vl + (h * (vl - vb) ** 2) / (2 * vl - 2 * vb)

    return result
