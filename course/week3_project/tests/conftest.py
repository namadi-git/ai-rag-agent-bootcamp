import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Ensure the project root (parent of tests/) is on sys.path so tests can import project modules
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))