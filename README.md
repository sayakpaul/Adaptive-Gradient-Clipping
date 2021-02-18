# Adaptive-Gradient-Clipping
Minimal implementation of adaptive gradient clipping (https://arxiv.org/abs/2102.06171) in TensorFlow 2. 

Adaptive gradient clipping is as follows (taken from ther paper) - 
$$
G_{i}^{\ell} \rightarrow\left\{\begin{array}{ll}
\lambda \frac{\left\|W_{i}^{\ell}\right\|_{E}^{\star}}{\left\|G_{i}^{\ell}\right\|_{F}} G_{i}^{\ell} & \text { if } \frac{\left\|G_{i}^{\ell}\right\|_{F}}{\left\|W_{i}^{\ell}\right\|_{F}^{\star}}>\lambda \\
G_{i}^{\ell} & \text { otherwise }
\end{array}\right.
$$

where, $\left\|W_{i}\right\|_{F}^{\star}=\max \left(\left\|W_{i}\right\|_{F}, \epsilon\right)$ and $\epsilon$ defaults to 1e-3.  