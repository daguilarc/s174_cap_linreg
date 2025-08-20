# Section 174 Capitalization Analysis Scripts

## Overview
These scripts analyze earnings risks from over-capitalization of R&D under recent changes to Section 174 of the Internal Revenue Code.

## Background
- **Tax Cuts and Jobs Act of 2017 (TCJA)**: Required non-GAAP capitalization of all R&D expenses
- **One Big Beautiful Bill Act of 2025 (OBBBA)**: Restored pre-TCJA full expensing status quo, but **ONLY for R&D activities recorded in the United States**
- **Foreign R&D Impact**: All R&D on foreign territory remains under the TCJA amortization schedule
- **Impact**: The book-to-tax income divergence is now irrelevant for US-based R&D tests, requiring a change to the R&D Deferral proxy ratio

## Scripts

### s174_obbba_ocf.py
- **Purpose**: Tests R&D deferral ratio impact on next period OCF margin
- **Independent Variable**: `rnd_deferral_ratio` (Capex / (Capex + (Operating Expenses - R&D Expenses)))
- **Dependent Variable**: `ocf_margin_next` (Next Period Operating Cash Flow Margin)
- **Use Case**: Analysis under OBBBA regime where full expensing is restored

### s174_tcja_ocf.py
- **Purpose**: Tests R&D to book-to-tax difference impact on next period OCF margin
- **Independent Variable**: `rnd_to_btd` (R&D expenses / Book-to-Tax Difference)
- **Dependent Variable**: `ocf_margin_next` (Next Period Operating Cash Flow Margin)
- **Use Case**: Analysis under TCJA regime where capitalization was required

### s174_tcja_earnings.py
- **Purpose**: Tests R&D to book-to-tax difference impact on earnings surprise ratio
- **Independent Variable**: `rnd_to_btd` (R&D expenses / Book-to-Tax Difference)
- **Dependent Variable**: `earnings_surprise_ratio` (Actual EPS / Estimated EPS)
- **Use Case**: Analysis under TCJA regime with earnings surprise as outcome

## Requirements

### Python Dependencies
- pandas
- numpy
- statsmodels
- scikit-learn
- matplotlib
- seaborn
- requests
- json
- os
- warnings

### API Requirements
- Financial Modeling Prep (FMP) API key
- Access to:
  - Income statements
  - Balance sheets
  - Cash flow statements
  - Earnings calendar (for earnings surprise analysis)
  - Company profiles
  - S&P 500 constituents

### Data Requirements
- S&P 500 company financial data
- Rolling 4-quarter financial metrics
- Sector and industry classifications for fixed effects
- Earnings estimates and actuals (for earnings surprise analysis)

### File Structure
- Cache files for API responses
- Output CSV files for analysis results
- Chart exports (PNG format)
- Diagnostic logs

## Usage
1. Set your FMP API key in the script
2. Run the desired analysis script
3. Review output files and charts
4. Analyze diagnostic statistics and model performance

## Output
- Regression analysis results
- VIF analysis for multicollinearity
- Outlier detection and filtering
- Cross-validation performance metrics
- Residual analysis
- Risk assessment for aggressive capitalization practices 
