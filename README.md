# DOE-Machine-Fit
A python script for making multi-dimentional reduced subset Design of Experiments(DOE) and fitting the data with Machine Learning(ML)

## Please Cite
[How To Optimize Materials and Devices via Design of Experiments and Machine Learning: Demonstration Using Organic Photovoltaics](https://pubs.acs.org/doi/full/10.1021/acsnano.8b04726)

## Example Code
```python
import pandas as pd
import matplotlib.pyplot as plt
import doe_machine_fit as doe

df = pd.read_csv("acsnano.csv")
var_prop_labels={'pce': 'Power Conversion Efficiency (%)',
        'don_con': 'wt% Donor Concentration',
        'spin_s': 'Spin Speed (rpm)',
        'total_con': 'Total Concentration (mg/ml)',
        }
doe.fit_svm(df, 'pce', ['don_con', 'spin_s', 'total_con'],gamma=0.15,
        mark_err=0.1, var_prop_labels=var_prop_labels)
plt.gcf().savefig("example_figure.png", dpi=300)
```

## Example Output
![Example graph using ascnano paper data](https://github.com/bcolsen/DOE-Machine-Fit/blob/main/acsnano_figure.png)
