@echo off
echo Installing Stock Market Prediction and Sentiment Analysis Project
echo =================================================================

echo Installing required Python packages...
pip install -r requirements.txt

echo Creating necessary directories...
if not exist "models" mkdir models
if not exist "output" mkdir output
if not exist "data" mkdir data

echo Installation complete!
echo.
echo You can now run the application using:
echo - run_app.bat       (for the Streamlit web interface)
echo - run_analysis.bat  (for command-line analysis)
echo.
echo Don't forget to configure your API keys in config.py before running!
pause 