CAPITALIZATION ANALYSIS SCRIPTS
==============================

This repository contains Python scripts for analyzing capitalization patterns in public companies using financial data from the Financial Modeling Prep API.

FILES INCLUDED:
- fmp_capitalization_analysis.py (Original version - S&P 500 companies)
- fmp_capitalization_analysis_v2.py (Modified version with different sector baseline - S&P 500 companies)
- fmp_capitalization_analysis_v3.py (Latest version - All US public companies)

OUTPUT FILES:
- capitalization_analysis_output.csv (Original version)
- capitalization_analysis_output_v2.csv (Modified version)
- capitalization_analysis_output_v3.csv (Latest version - all US public companies)
- rnd_deferral_vs_ocf_margin.png (Original version)
- rnd_deferral_vs_ocf_margin_v2.png (Modified version)
- rnd_deferral_vs_ocf_margin_v3.png (Latest version)

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

Run the original version (S&P 500):
```bash
python3 fmp_capitalization_analysis.py
```

Run the modified version (S&P 500, v2):
```bash
python3 fmp_capitalization_analysis_v2.py
```

Run the latest version (All US public companies, v3):
```bash
python3 fmp_capitalization_analysis_v3.py
```

KEY DIFFERENCES BETWEEN VERSIONS:
================================

Original Version (fmp_capitalization_analysis.py):
- Scope: S&P 500 companies only
- Baseline sector: "Software"
- All software companies (both applications and infrastructure) are in the baseline
- cap_intensity is a control variable (not a treatment variable)
- Includes 19 control variables with rolling 4-quarter calculations

Modified Version (fmp_capitalization_analysis_v2.py):
- Scope: S&P 500 companies only
- Baseline sector: "Software - Applications"
- "Software - Infrastructure" companies are moved to "Technology Excluding Software" dummy variable
- cap_intensity is a control variable (not a treatment variable)
- Includes 19 control variables with rolling 4-quarter calculations

Latest Version (fmp_capitalization_analysis_v3.py):
- Scope: All US public companies (expanded from S&P 500)
- Baseline sector: "Software - Applications"
- References to "S&P 500" replaced with "Market" in charts and output
- cap_intensity is a control variable (not a treatment variable)
- Includes 19 control variables with rolling 4-quarter calculations
- Enhanced data collection for broader company universe

MAJOR UPDATES IN V3:
===================

1. **Expanded Scope**: Now analyzes all US public companies instead of just S&P 500
2. **Market Cap Control**: Added log_market_cap as a control variable
3. **Total Assets Control**: Added log_total_assets as a control variable
4. **D&A Expense**: Added total_dna ratio (D&A expense / total assets) as control variable
5. **PPE Calculation**: Changed from rolling 4-quarter average to rolling 4-quarter delta
6. **Chart Updates**: Removed cap_intensity chart (now only R&D deferral ratio chart)
7. **Output Naming**: All outputs renamed with "_v3" suffix

CONTROL VARIABLES INCLUDED:
==========================

The regression model includes 19 control variables:

1. **cap_intensity** (now a control variable): Capex / R&D Expense
2. **rnd_deferral_ratio** (primary treatment variable): Capex / [Capex + (Operating Expenses - R&D Expense)]
3. **leverage**: Total Debt / Total Assets
4. **revenue_growth**: Quarter-over-quarter percentage change
5. **current_inventory_ratio**: Inventory / Total Current Assets
6. **ocf_margin_lagged**: Previous period OCF margin
7. **profitability**: Gross Profit / Total Assets
8. **ppe_total_assets**: ΔPPE / Total Assets
9. **rnd_intensity**: R&D Expense / Revenue
10. **acquisitions_intensity**: |Acquisitions Net| / Total Assets
11. **goodwill_change_intensity**: |ΔGoodwill| / Intangible Assets
12. **intangible_assets_change_intensity**: ΔIntangible Assets / Total Assets
13. **total_dna**: Depreciation & Amortization / Total Assets
14. **log_market_cap**: ln(Market Capitalization)
15. **log_total_assets**: ln(Total Assets)
16. **earnings_quality**: Operating Cash Flow / Net Income
17. **quarter_number**: Fiscal quarter (1-4)
18. **year**: Calendar year
19. **Industry Fixed Effects**: Sector dummy variables

ROLLING 4-QUARTER CALCULATIONS:
==============================

The model uses three types of rolling 4-quarter calculations:

1. **Rolling 4-Quarter Sum**: For flow variables (revenues, expenses, cash flows)
   - Formula: Xₜˢᵘᵐ = Xₜ + Xₜ₋₁ + Xₜ₋₂ + Xₜ₋₃

2. **Rolling 4-Quarter Average**: For stock variables (assets, debt, equity)
   - Formula: Yₜᵃᵛᵍ = (Yₜ + Yₜ₋₁ + Yₜ₋₂ + Yₜ₋₃) / 4

3. **Rolling 4-Quarter Delta**: For change variables (PPE changes, goodwill changes)
   - Formula: Zₜᵈᵉˡᵗᵃ = Zₜ - Zₜ₋₄

WHAT THE SCRIPTS DO:
===================

1. Fetch financial data for companies from Financial Modeling Prep API
2. Calculate capitalization intensity and R&D deferral ratios
3. Apply rolling 4-quarter calculations to transform quarterly data
4. Perform regression analysis with industry fixed effects
5. Generate scatter plots showing relationships between:
   - R&D deferral ratio vs. next period OCF margin (primary treatment variable)
6. Export results to CSV and PNG files
7. Perform multicollinearity checks (VIF analysis)
8. Conduct cross-validation for model robustness

DATA SOURCES:
============

The scripts fetch the following financial data:
- Balance Sheet data
- Income Statement data
- Cash Flow Statement data
- Company profiles and sector information
- Market capitalization data
- Quotes and pricing data

CACHE SYSTEM:
============

The scripts use a caching system to avoid repeated API calls:
- Cache files are stored as JSON files
- Cache expires after 30 days
- Cache files are automatically created and updated
- Includes separate caches for different data types

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

5. Large dataset processing (v3)
   Solution: v3 may take longer due to expanded scope; ensure sufficient RAM (4-8GB recommended)

PERFORMANCE NOTES:
=================

- First run may take 15-30 minutes due to initial data fetching
- v3 version may take longer due to expanded company universe
- Subsequent runs are faster due to caching
- Memory usage: ~2-8 GB RAM recommended (higher for v3)
- Disk space: ~1-2GB for cache files and outputs (higher for v3)

OUTPUT INTERPRETATION:
=====================

The scripts generate:
1. Regression analysis results with statistical significance
2. Industry fixed effects coefficients
3. Multicollinearity diagnostics (VIF analysis)
4. Cross-validation results
5. Visual charts showing relationships between variables
6. Comprehensive control variable analysis

For detailed interpretation of results, refer to the statistical output and diagnostic messages printed during execution.

REGRESSION MODEL:
================

The complete regression model is:
OCF Marginₜ₊₁ = β₀ + β₁ Cap Intensityₜ + β₂ R&D Deferral Ratioₜ + Σᵢ₌₃¹⁸ βᵢ Control Variableᵢ + Σⱼ₌₁ᵏ γⱼ Sectorⱼ + εₜ

Where:
- OCF Marginₜ₊₁ is the next period's operating cash flow margin
- All independent variables are measured at time t
- εₜ is the error term
- βᵢ and γⱼ are regression coefficients

SUPPORT:
========

For issues with the Financial Modeling Prep API, contact their support.
For script-specific issues, check the error messages and diagnostic output.

LICENSE:
========

This code is provided as-is for educational and research purposes. 