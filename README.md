## Change Point Enhanced Anomaly Detection for IoT Time Series Data


Prerequisite
* Python 3.7.x
* numpy (latest version)
* pandas (latest version)
* matplotlib (latest version)
* ruptures (latest version)
* pyod (latest version)
* scipy (latest version)
* scikit-learn (latest version) 

For anomaly detection use the ``ad.py`` script

``
python ad.py example.csv 
``


For change point detection use the ``cpd.py`` script 

``
python cdp.py example.csv 
``

For the decision use the ``decision.py`` script with the output of the other 2 scripts

``
python decision.py results_ad.csv results_cpd.csv
``

## Article:

Elena-Simona Apostol, Ciprian-Octavian Truică, Florin Pop and Christian Esposito. *Change Point Enhanced Anomaly Detection for IoT Time Series Data*. In WATER, 13(12):1-19(1633), ISSN 2073-4441, MDPI, June 2021. DOI: [10.3390/w13121633 ](http://doi.org/10.3390/w13121633 )
