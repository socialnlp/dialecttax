from . import graders
from . import multivalue
from . import parallelaave
from . import redial

DATASET_MODULES = {
    "redial": redial,
    "parallelaave": parallelaave,
    "multivalue": multivalue,
}
