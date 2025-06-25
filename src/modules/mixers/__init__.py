from .vdn import VDNMixer
from .nmix import Mixer
from .qatten import QattenMixer
from .avdn import AVDNMixer

REGISTRY = {}

REGISTRY["vdn"] = VDNMixer
REGISTRY["nmix"] = Mixer
REGISTRY["qatten"] = QattenMixer
REGISTRY["avdn"] = AVDNMixer
