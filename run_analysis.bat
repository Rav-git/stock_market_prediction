@echo off
echo Stock Market Prediction with Sentiment Analysis
echo =============================================

REM Use default parameters
if "%1"=="" (
    echo Running default analysis for AAPL...
    python src/main.py --symbol AAPL
) else (
    echo Running analysis for %1...
    python src/main.py --symbol %1 %2 %3 %4 %5 %6 %7 %8 %9
)
pause 