import click
import sys
from pathlib import Path

import tskit
import tsinfer
import cyvcf2

import numpy as np

sys.path.append(Path(sys.path[0]) / "python")
import util


# Get .samples file from VCF file
# Get variant statistics from the .samples file
