# CampusMind 🎓

## Setup
1. Clone repo
2. Create virtual environment
   python -m venv .venv
3. Install dependencies
   pip install -r requirements.txt
4. Create .env file and add
   GROQ_API_KEY=your_key_here
5. Run ingestion
   python ingest.py
6. Run app
   streamlit run app.py