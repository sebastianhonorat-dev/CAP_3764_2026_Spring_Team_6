import os
from dotenv import load_dotenv
from pathlib import Path

root = Path(__file__).resolve().parents[3]
os.chdir(root)

load_dotenv()

SCORECARD_KEY = os.getenv("COLLEGE_SCORECARD_API_KEY")

BASE_URL = "https://api.data.gov/ed/collegescorecard/v1/schools"