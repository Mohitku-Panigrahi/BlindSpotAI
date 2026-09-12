Clinical Blindspot AI - Shareable Project Bundle

Contents
- clinical_blindspot.py               Backend AI pipeline + API server
- requirements.txt                    Python dependencies
- setup_python_3_14_3.bat             Python version setup helper
- setup_whisper_model.bat              Whisper model local downloader
- launch_clinical_app.bat             One-click start script
- stop_clinical_app.bat               One-click stop script
- frontend\index.html                 Intro page
- frontend\mediai-v3.html             Main frontend app
- frontend\welcome_intro.mp4          Intro media

Quick start (Windows)
1) Install Python 3.14.3 (required).
2) If unsure, run setup_python_3_14_3.bat first.
3) Double-click launch_clinical_app.bat.
4) Wait for browser to open automatically at:
   http://127.0.0.1:5500/index.html
5) Login:
   doctor / doctor123
   nurse  / nurse123
   admin  / admin123

Python 3.14.3 setup steps
1) Download installer: https://www.python.org/downloads/release/python-3143/
2) Run installer and enable:
   - Add Python to PATH
   - Install launcher for all users (recommended)
3) Restart terminal after install.
4) Verify with command:
   py -3.14 --version

What the launcher does
- Installs dependencies from requirements.txt (pip)
- Starts backend API on port 8000
- Starts frontend static server on port 5500
- Opens the app in your browser

About VAD on Python 3.14
- webrtcvad-wheels can fail to build on some Python 3.14 setups.
- The app now includes an automatic energy-based VAD fallback.
- So the project runs even if webrtcvad is not installed.

Whisper model setup (important for offline PCs)
1) On a machine with internet, run setup_whisper_model.bat.
2) This creates models\faster-whisper-base.
3) Copy that folder into your project on offline machine at:
   models\faster-whisper-base
4) Launch app normally with launch_clinical_app.bat.

If you see LocalEntryNotFoundError or getaddrinfo failed
- It means model download from Hugging Face failed (internet/DNS issue).
- Fix by using the local model copy method above.

How to stop
- Double-click stop_clinical_app.bat

API base used by frontend
- http://127.0.0.1:8000
- Web UI supports local browser audio recording and backend analysis

Notes
- First launch may take longer due to dependency/model setup.
- If a firewall prompt appears, allow local access.
- GPU is optional; backend auto-falls back to CPU if CUDA is unavailable.
- Launcher enforces Python 3.14.x and will stop with guidance if missing.

If widget shows offline on another PC
1) Run launch_clinical_app.bat and keep both opened terminals running.
2) In browser, open http://127.0.0.1:8000/health
   - Expected response: {"status":"ok", ...}
3) If /health fails, backend did not start; check "Clinical Backend" terminal output.
4) Allow Python/PowerShell through firewall prompts.
5) Ensure API Base URL in widget is exactly http://127.0.0.1:8000
