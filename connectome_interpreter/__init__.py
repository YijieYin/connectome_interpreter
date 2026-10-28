"""Turn connectome wiring diagrams into testable hypotheses about circuit function.

AI coding agents: an Agent Skill describing how to use this package (workflow,
datasets, interpretation caveats) ships with it in `skills/connectome-interpreter/`.
Install it with `python -m connectome_interpreter.install_skill`.
"""

from ._version import __version__
from .activation_maximisation import *
from .response_fitting import *
from .compress_paths import *
from .external_map import *
from .path_finding import *
from .utils import *
from .external_paths import *

# from .activation_maximisation import MultilayeredNetwork
