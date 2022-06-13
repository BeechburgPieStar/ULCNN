# Ultra-Lite-Convolutional-Neural-Network-for-Automatic-Modulation-Classification

In this paper, we designed a ultra lite CNN for AMC, and its simulation is based on RML2016A


$\begin{table*}
  \caption{The average classification performances.}
  \begin{center}
  \begin{tabular}{|c|c|c|c|c|c|c|c|c|}
  \hline
  $R_{sample}$&Original&Rotate&Flip&Gaussian&CS&CSoB (proposed)&CutMix&VAT\\
  \hline
  1\% (770 samples)&	21.29\%	&	30.36\%	&	20.85\%	&	20.81\%	&	29.29\%	&	\textbf{41.95\%}	&	26.96\%	&	13.81\%	\\
  \hline
  2.5\% (1,925 samples)&	29.30\%	&	32.78\%	&	29.17\%	&	25.26\%	&	38.34\%	&	\textbf{45.05\%}	&	37.68\%	&	13.81\%	\\
  \hline
  5\% (3,850 samples)&	34.29\%	&	38.33\%	&	31.08\%	&	31.35\%	&	39.67\%	&	\textbf{49.04\%}	&	40.18\%	&	14.58\%	\\
  \hline
  10\% (7,700 samples)&	39.61\%	&	45.91\%	&	41.68\%	&	34.31\%	&	47.17\%	&	\textbf{51.39\%}	&	42.54\%	&	14.47\%	\\
  \hline
  100\% (77,000 samples)&	52.03\%	&	\textbf{53.99\%}	&	51.60\%	&	49.97\%	&	53.02\%	&	53.65\%	&	51.22\%	&	22.55\%	\\
  \hline
  \end{tabular}
  \end{center}
  \label{Tab:Structure_FLOPs_Param2}
\end{table*}$
