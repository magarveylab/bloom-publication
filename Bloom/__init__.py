import os

from dotenv import load_dotenv

curdir = os.path.abspath(os.path.dirname(__file__))
dataset_dir = f"{curdir}/datasets"
dotenv_path = os.path.join(curdir, ".env")
load_dotenv(dotenv_path)
