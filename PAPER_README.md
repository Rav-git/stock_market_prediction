# Stock Market Prediction Research Paper

This directory contains the files for the research paper "Stock Market Prediction Using Sentiment Analysis and Machine Learning: A Hybrid Approach" formatted according to Elsevier's article template.

## Files Included

- `Stock_Market_Prediction_Research_Paper.tex` - Main LaTeX file for the paper
- `references.bib` - BibTeX file containing all references
- `system_architecture.txt` - Text description of Figure 1 (system architecture diagram)

## How to Compile

The paper can be compiled using any LaTeX distribution that supports the Elsevier article class (`elsarticle`). Follow these steps:

1. Ensure you have a LaTeX distribution installed (e.g., TeXLive, MiKTeX)
2. Install the required package `elsarticle` if not already included
3. Compile the document using the following commands:

```bash
pdflatex Stock_Market_Prediction_Research_Paper.tex
bibtex Stock_Market_Prediction_Research_Paper
pdflatex Stock_Market_Prediction_Research_Paper.tex
pdflatex Stock_Market_Prediction_Research_Paper.tex
```

This will generate the PDF file `Stock_Market_Prediction_Research_Paper.pdf`.

## Creating Figures

The paper mentions several figures:

1. **Figure 1**: System architecture diagram - Use the description in `system_architecture.txt` to create a diagram using your preferred diagramming tool (e.g., Microsoft Visio, draw.io, OmniGraffle).

2. **Figure 2**: Sentiment-price correlation chart - This should be created based on actual data from the project showing the relationship between sentiment scores and stock price movements for AAPL over a 3-month period.

## Required LaTeX Packages

The paper uses the following LaTeX packages that should be installed:
- `elsarticle` (Elsevier article class)
- `lineno` (for line numbering)
- `hyperref` (for hyperlinks)
- `graphicx` (for including graphics)
- `amssymb` and `amsmath` (for math symbols and equations)
- `algorithm` and `algorithmic` (for algorithm descriptions)
- `booktabs` (for high-quality tables)
- `tabularx` (for advanced tables)
- `float` (for better figure placement)
- `listings` (for code listings)
- `xcolor` (for colored text)

## Note on Figure Creation

To create the actual figures for the paper, you would need to:

1. Convert the system architecture text description into a proper diagram
2. Generate visualizations from the actual project data showing:
   - Sentiment vs. price correlations
   - Prediction accuracy comparisons
   - Model performance by stock

## Contact

For any questions or clarifications about this research paper, please contact the corresponding author. 