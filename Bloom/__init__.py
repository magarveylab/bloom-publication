import os

from dotenv import load_dotenv

from Bloom.BloomRXN.utils import curdir

dotenv_path = os.path.join(curdir, ".env")
load_dotenv(dotenv_path)
