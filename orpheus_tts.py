from scipy.io.wavfile import write
from orpheus_cpp import OrpheusCpp
import numpy as np

orpheus = OrpheusCpp(verbose=False)

text = "Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot."
buffer = []
for i, (sr, chunk) in enumerate(orpheus.stream_tts_sync(text, options={"voice_id": "tara"})):
   buffer.append(chunk)
   print(f"Generated chunk {i}")
buffer = np.concatenate(buffer, axis=1)
write("output.wav", 24_000, np.concatenate(buffer))