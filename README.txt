CAPITALIZATION ANALYSIS SCRIPTS
==============================

This repository contains Python scripts for analyzing capitalization patterns in S&P 500 companies using financial data from the Financial Modeling Prep API.

FILES INCLUDED:
- fmp_capitalization_analysis.py (Original version)
- fmp_capitalization_analysis_v2.py (Modified version with different sector baseline)

OUTPUT FILES:
- capitalization_analysis_output.csv (Original version)
- capitalization_analysis_output_v2.csv (Modified version)
- cap_intensity_vs_ocf_margin.png (Original version)
- cap_intensity_vs_ocf_margin_v2.png (Modified version)
- rnd_deferral_vs_ocf_margin.png (Original version)
- rnd_deferral_vs_ocf_margin_v2.png (Modified version)

PREREQUISITES:
=============

1. Python 3.7 or higher
2. Financial Modeling Prep API key (free tier available)

REQUIRED PYTHON PACKAGES:
========================

Install the following packages using pip:

```bash
pip3 install pandas
pip3 install numpy
pip3 install matplotlib
pip3 install seaborn
pip3 install statsmodels
pip3 install scikit-learn
pip3 install requests
```

Or install all at once:

```bash
pip3 install pandas numpy matplotlib seaborn statsmodels scikit-learn requests
```

API KEY SETUP:
=============

1. Sign up for a free account at: https://financialmodelingprep.com/
2. Get your API key from your account dashboard
3. The script will prompt you to enter the API key when first run

USAGE:
======

Run the original version:
```bash
python3 fmp_capitalization_analysis.py
```

Run the modified version (v2):
```bash
python3 fmp_capitalization_analysis_v2.py
```

KEY DIFFERENCES BETWEEN VERSIONS:
================================

Original Version (fmp_capitalization_analysis.py):
- Baseline sector: "Software"
- All software companies (both applications and infrastructure) are in the baseline

Modified Version (fmp_capitalization_analysis_v2.py):
- Baseline sector: "Software - Applications"
- "Software - Infrastructure" companies are moved to "Technology Excluding Software" dummy variable

WHAT THE SCRIPTS DO:
===================

1. Fetch financial data for S&P 500 companies from Financial Modeling Prep API
2. Calculate capitalization intensity and R&D deferral ratios
3. Perform regression analysis with industry fixed effects
4. Generate scatter plots showing relationships between:
   - Capitalization intensity vs. next period OCF margin
   - R&D deferral ratio vs. next period OCF margin
5. Export results to CSV and PNG files

DATA SOURCES:
============

The scripts fetch the following financial data:
- Balance Sheet data
- Income Statement data
- Cash Flow Statement data
- Company profiles and sector information

CACHE SYSTEM:
============

The scripts use a caching system to avoid repeated API calls:
- Cache files are stored as JSON files
- Cache expires after 30 days
- Cache files are automatically created and updated

TROUBLESHOOTING:
===============

Common Issues:

1. "ModuleNotFoundError: No module named 'pandas'"
   Solution: Install required packages using pip3

2. API rate limiting
   Solution: The scripts include built-in rate limiting and retry logic

3. Memory issues with large datasets
   Solution: The scripts include robust data filtering and outlier detection

4. Network connectivity issues
   Solution: Check your internet connection and API key validity

PERFORMANCE NOTES:
=================

- First run may take 10-15 minutes due to initial data fetching
- Subsequent runs are faster due to caching
- Memory usage: ~2-4 GB RAM recommended
- Disk space: ~500MB for cache files and outputs

OUTPUT INTERPRETATION:
=====================

The scripts generate:
1. Regression analysis results with statistical significance
2. Industry fixed effects coefficients
3. Multicollinearity diagnostics (VIF analysis)
4. Cross-validation results
5. Visual charts showing relationships between variables

For detailed interpretation of results, refer to the statistical output and diagnostic messages printed during execution.

SUPPORT:
========

For issues with the Financial Modeling Prep API, contact their support.
For script-specific issues, check the error messages and diagnostic output.

LICENSE:
========

This code is provided as-is for educational and research purposes. 