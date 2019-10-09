# timeseries-forecasting-anomalydetection
This repo utilizes fbprophet for forecasting and RPCA for anomaly detection on time-series data.

Three classes are implemented.  The first class is to retrieve data and prepare it for analysis and forecasting.  The second class utilizes the facebook Prophet package to do forecasting. Visualizations are implemented as part of the method arguments to provide easy to iterate process.  The thirs class utlizes the robust PCA algorithms based on Principal Components Pursuit (PCP) https://en.wikipedia.org/wiki/Robust_principal_component_analysis.  
The PCP method implementation has been borrowed from https://github.com/dfm/pcp. The demo provided by Dan Foreman-Mackey demonstrates the usage of RPCA to extract background from the images. In this implementation, RPCA is utilized to identify anomalies through the extraction of Sparse matrix. 

