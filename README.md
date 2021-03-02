# Matlab_Code
Matlab codes
This is Matlab codes for logistic regression implement.

Loss function:
\begin{subequations}
\begin{align*} 
L(\theta) &= -\frac{1}{N} \sum_{n=1}^{N}\ln{\frac{1}{1+e^{-(2y_n-1)z_n}}}\\
&= \frac{1}{N}\sum_{n=1}^{N} \zeta((1-2y_n)z_n),
\end{align*}
\end{subequations}
