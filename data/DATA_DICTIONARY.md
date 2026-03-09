# Data Dictionary

**Dataset:** Default of Credit Card Clients (UCI ML Repository, ID: 350)
**Source:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
**Records:** 30,000 | **Features:** 23 + 1 target | **Period:** April–September 2005 (Taiwan)

---

## Target Variable

| Variable | UCI Name | Description | Values |
|---|---|---|---|
| `DEFAULT` | `Y` | Default payment next month | 0 = No, 1 = Yes |

## Demographic Features

| Variable | UCI Name | Description | Values / Unit |
|---|---|---|---|
| `LIMIT_BAL` | `X1` | Amount of given credit (individual + family supplementary credit) | NT dollar |
| `SEX` | `X2` | Gender | 1 = Male, 2 = Female |
| `EDUCATION` | `X3` | Education level | 1 = Graduate school, 2 = University, 3 = High school, 4 = Other* |
| `MARRIAGE` | `X4` | Marital status | 1 = Married, 2 = Single, 3 = Other* |
| `AGE` | `X5` | Age | Years |

*\*Undocumented values (EDUCATION: 0, 5, 6; MARRIAGE: 0) merged into "Other" during cleaning.*

## Repayment Status (History of Past Payment)

Measurement scale: -2 = no consumption; -1 = pay duly; 0 = use of revolving credit; 1 = payment delay 1 month; 2 = delay 2 months; ...; 8 = delay 8 months; 9 = delay 9+ months.

| Variable | UCI Name | Description | Month |
|---|---|---|---|
| `PAY_0` | `X6` | Repayment status in September 2005 | Sep 2005 |
| `PAY_2` | `X7` | Repayment status in August 2005 | Aug 2005 |
| `PAY_3` | `X8` | Repayment status in July 2005 | Jul 2005 |
| `PAY_4` | `X9` | Repayment status in June 2005 | Jun 2005 |
| `PAY_5` | `X10` | Repayment status in May 2005 | May 2005 |
| `PAY_6` | `X11` | Repayment status in April 2005 | Apr 2005 |

## Bill Statement Amount

| Variable | UCI Name | Description | Month |
|---|---|---|---|
| `BILL_AMT1` | `X12` | Bill statement amount (NT dollar) | Sep 2005 |
| `BILL_AMT2` | `X13` | Bill statement amount (NT dollar) | Aug 2005 |
| `BILL_AMT3` | `X14` | Bill statement amount (NT dollar) | Jul 2005 |
| `BILL_AMT4` | `X15` | Bill statement amount (NT dollar) | Jun 2005 |
| `BILL_AMT5` | `X16` | Bill statement amount (NT dollar) | May 2005 |
| `BILL_AMT6` | `X17` | Bill statement amount (NT dollar) | Apr 2005 |

## Previous Payment Amount

| Variable | UCI Name | Description | Month |
|---|---|---|---|
| `PAY_AMT1` | `X18` | Amount paid (NT dollar) | Sep 2005 |
| `PAY_AMT2` | `X19` | Amount paid (NT dollar) | Aug 2005 |
| `PAY_AMT3` | `X20` | Amount paid (NT dollar) | Jul 2005 |
| `PAY_AMT4` | `X21` | Amount paid (NT dollar) | Jun 2005 |
| `PAY_AMT5` | `X22` | Amount paid (NT dollar) | May 2005 |
| `PAY_AMT6` | `X23` | Amount paid (NT dollar) | Apr 2005 |

## Engineered Features (created in `src/data_prep.py`)

| Variable | Formula | Description |
|---|---|---|
| `UTILISATION_RATIO` | mean(BILL_AMT1–6) / LIMIT_BAL | Average credit utilisation intensity |
| `AVG_PAY_AMT` | mean(PAY_AMT1–6) | Average monthly payment amount |
| `MAX_DELAY` | max(PAY_0, PAY_2–6) | Worst repayment delay across all 6 months |
| `PAY_BILL_RATIO` | mean(PAY_AMT) / mean(BILL_AMT) | Ratio of payments to bills (repayment discipline) |

---

## Reference

Yeh, I.-C. and Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), pp. 2473–2480.
