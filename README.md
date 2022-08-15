# SPreCC
**S**imilarity-based **Pre**diction of **C**rystallization **C**onditions

## Overview

Knowledge of the three-dimensional structures of proteins is essential for understanding their biological functions and developing targeted molecular therapies.  The most common technique for elucidating the structure of a protein is X-ray crystallography, but obtaining high quality crystals for X-ray diffraction remains the bottleneck of this technique.  Predicting chemical conditions that would produce high quality crystals from the amino acid sequence of a protein would considerably speed this process.  Previous efforts have attempted to determine the optimal pH for crystallization from a protein's amino acid sequence and its pI or predict a list of chemical conditions for crystallization from its amino acid sequence.  However, to our knowledge, no attempt has been made to predict the technique, pH, temperature, chemical conditions, and concentrations of those chemical conditions all together from a protein's amino acid sequence.  Here, we present and evaluate SPreCC, a method of predicting categorical, continuous, and presence/absence information for crystalliation conditions based on amino acid sequence similarity.

## Introduction

## Models

We will present four models for different types of crystallization condition data.  The first will be a model for predicting the presence/absence status of chemical conditions such as sodium chloride or tris.  The second will be a model for predicting categorical values, which will be applied only to predicting the crystallization technique to be used.  The third will be a model for predicting concentrations of chemical conditions like the molarity of sodium chloride or the percent volume per volume of dimethyl sulfoxide.  The fourth will be a model for simultaneously predicting concentrations and polymer lengths for chemical conditions like polyethylene glycol (PEG).  A single condition might use multiple models; for example, sodium chloride uses both the presence/absence model for predicting whether it should be in the crystallization mixture and the concentration model for predicting its optimal concentration if it is present.  In general, sequence similarities will be determined using [Mash](https://mash.readthedocs.io/en/latest/index.html), and protein indexing will be over proteins with the relevant data available from the [Protein Data Bank](https://www.rcsb.org/).

### Model 1 (presence/absence)

Let $y$ be the true binary value we are interested in (such as whether or not sodium chloride is present).  Let our prediction for the binary value be $\hat{y}$ such that 

$$\hat{y} = \frac{\bar{x}}{n_p+1} + \frac{n_p}{n_p+1}\cdot\frac{\sum\limits_{i} x_i\sigma(w_1s_i+w_0)}{\sum\limits_{i} \sigma(w_1s_i+w_0)} \textrm{ for } i \textrm{ such that } p_i<\tau$$

 where $x_i$ is the binary value for protein $i$ (e.g. whether sodium chloride was present for protein $i$), $s_i\in[0,1]$ is 1 minus the [Mash](https://mash.readthedocs.io/en/latest/index.html) distance between protein $i$ and the protein of interest, $\sigma$ is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function), $p_i$ is the Mash p-value of the similarity between protein $i$ and the target protein (how likely the two proteins are to have their reported degree of similarity by chance), $\bar{x}$ is the average value of the binary condition across all the dataset excluding the protein of interest, and $n_p$ is the total number of proteins with Mash p-values less than $\tau$.  Intuitively, we are taking a weighted average between the binary values from all the proteins and the binary values from related proteins.  This ensures that the model still gives an estimate for proteins with no similar proteins in the database while also allowing predictions for proteins with even a few similar proteins to be mostly determined by the conditions of those similar proteins.  Within the term corresponding to similar proteins, $\sigma(w_1s_i+w_0)$ is the weight for the crystallization condition of protein $i$, and the denominator normalizes the calculation.  Each weight should be some value between 0 and 1, and we expect greater sequence identities to correspond to heavier weights, but the model allows flexibility in how much some amount of additional sequence identity should increase the weight.  This weighting scheme allows much more flexibility and speed than, for example, incorporating the distance of every protein or ranking the most similar proteins.  It allows a variable number of inputs, preventing a need for as many independent weights as there are proteins, and it allows the weight to be determined directly from the sequence similarity rather than from some ranking of similarities.  We will attempt to minimize the negative log-likelihood loss: $L(y,\hat{y})=-[y\ln(\hat{y}) + (1-y)\ln(1-\hat{y})]$.

The specified model enables the fitting of two parameters: $w_0$ and $w_1$.  Let $\sigma_i=\sigma(w_1s_i+w_0)$.  Applying the chain rule, we obtain the following:

$$\begin{align*} 
\frac{\partial L(\hat{y},y)}{\partial w_0} &= \frac{\partial L(\hat{y},y)}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_0}\\ 
&= -\left[\frac{y}{\hat{y}}-\frac{1-y}{1-\hat{y}}\right]\cdot \frac{\left(\sum\limits_{i}\sigma_i\right)\left(\sum\limits_{i}x_i\sigma_i(1-\sigma_i)\right)-\left(\sum\limits_{i}x_i\sigma_i\right)\left(\sum\limits_{i}\sigma_i(1-\sigma_i)\right)}{\left(\sum\limits_{i}\sigma_i\right)^2}\\
\frac{\partial L(\hat{y},y)}{\partial w_1} &=  \frac{\partial L(\hat{y},y)}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_1}\\ 
&=  -\left[\frac{y}{\hat{y}}-\frac{1-y}{1-\hat{y}}\right]\cdot \frac{\left(\sum\limits_{i}\sigma_i\right)\left(\sum\limits_{i}x_i\sigma_i(1-\sigma_i)s_i\right)-\left(\sum\limits_{i}x_i\sigma_i\right)\left(\sum\limits_{i}\sigma_i(1-\sigma_i)s_i\right)}{\left(\sum\limits_{i}\sigma_i\right)^2}
\end{align*}$$

Because of the memory requirements involved in manipulating all the amino acid identity scores at once, we will use stochastic gradient descent to pick a protein at random, determine its amino acid identity against all the other proteins, predict its condition, and update the weights according to the loss.  With a learning rate $\alpha$, the update statements will be as follows:

$$\begin{align*} 
w_0&\leftarrow w_0-\alpha\frac{\partial L(\hat{y},y)}{\partial w_0}\\
w_1&\leftarrow w_1-\alpha\frac{\partial L(\hat{y},y)}{\partial w_1}
\end{align*}$$

To achieve an initially plausible weight scheme, the following initializations will chosen: $w_0=-1$, $w_1=3$.  The following image shows the weights produced by these parameters.

<p align="center">
<img src="images/binary.png" width="300" height="200">
</p>

The learning rate will start as $\alpha=0.1$ and will decay by $1/((\textrm{number of proteins}) \cdot (\textrm{number of epochs}))$ after each update.  The Mash p-value threshold will be $\tau=1/(\textrm{number of proteins})\approx 9\cdot10^{-6}$.

### Model 2 (classification)

Let $y$ be the true multi-class value we are interested in (such as whether a protein requires vapor diffusion, lipidic cubic phase, etc.) where $y$ is a one-hot vector with each element representing one class.  Let our prediction for the multiclass value be $\hat{y}$ such that 

$$\hat{y} = \frac{\bar{x}}{n_p+1} + \frac{n_p}{n_p+1}\cdot\frac{\sum\limits_{i} x_i\sigma(w_1s_i+w_0)}{\sum\limits_{i} \sigma(w_1s_i+w_0)} \textrm{ for } i \textrm{ such that } p_i<\tau$$

 where $x_i$ is the one-hot encoded vector for protein $i$, $s_i\in[0,1]$ is 1 minus the Mash distance between protein $i$ and the protein of interest, $\sigma$ is the sigmoid function, $p_i$ is the Mash p-value of the similarity between protein $i$ and the target protein, $\bar{x}$ is the element-wise average of the one-hot encoded vectors across all the dataset excluding the protein of interest, and $n_p$ is the total number of proteins with Mash p-values less than $\tau$.  Intuitively, we are taking a weighted average between the classes of all the proteins and the classes of related proteins.  This ensures that the model still predicts a class for proteins with no similar proteins in the database while also allowing predictions for proteins with even a few similar proteins to be mostly determined by the classes of those similar proteins.  Within the term corresponding to similar proteins, $\sigma(w_1s_i+w_0)$ is the weight for the class of protein $i$, and the denominator normalizes the calculation.  We will attempt to minimize the negative log-likelihood loss: 
 
$$L(y,\hat{y})=-\sum_{k=1}^K y_k \ln (\hat{y}_k) $$ 

where $K$ is the number of classes.

The specified model enables the fitting of two parameters: $w_0$ and $w_1$.  Because of the loss specification, the gradient with respect to $w_0$ or $w_1$ will only pass through the chain rule with the $\hat{y}_k$ that corresponds to the correct $y_k$.  Let $\sigma_i=\sigma(w_1s_i+w_0)$.  Applying the chain rule, we obtain the following:

$$\begin{align*} 
\frac{\partial L(\hat{y},y)}{\partial w_0} &= \frac{\partial L(\hat{y},y)}{\partial \hat{y_k}} \cdot \frac{\partial \hat{y_k}}{\partial w_0}\\ 
&= -\left[\frac{1}{\hat{y_k}}\right]\cdot \frac{\left(\sum\limits_{i}\sigma_i\right)\left(\sum\limits_{i}x_{k,i}\sigma_i(1-\sigma_i)\right)-\left(\sum\limits_{i}x_{k,i}\sigma_i\right)\left(\sum\limits_{i}\sigma_i(1-\sigma_i)\right)}{\left(\sum\limits_{i}\sigma_i\right)^2}\\
\frac{\partial L(\hat{y},y)}{\partial w_1} &=  \frac{\partial L(\hat{y},y)}{\partial \hat{y_k}} \cdot \frac{\partial \hat{y_k}}{\partial w_1}\\ 
&=  -\left[\frac{1}{\hat{y_k}}\right]\cdot \frac{\left(\sum\limits_{i}\sigma_i\right)\left(\sum\limits_{i}x_{k,i}\sigma_i(1-\sigma_i)s_i\right)-\left(\sum\limits_{i}x_{k,i}\sigma_i\right)\left(\sum\limits_{i}\sigma_i(1-\sigma_i)s_i\right)}{\left(\sum\limits_{i}\sigma_i\right)^2}
\end{align*}$$

Because of the memory requirements involved in manipulating all the amino acid identity scores at once, we will use stochastic gradient descent to pick a protein at random, determine its amino acid identity against all the other proteins, predict its condition, and update the weights according to the loss.  With a learning rate $\alpha$, the update statements will be as follows:

$$\begin{align*} 
w_0&\leftarrow w_0-\alpha\frac{\partial L(\hat{y},y)}{\partial w_0}\\
w_1&\leftarrow w_1-\alpha\frac{\partial L(\hat{y},y)}{\partial w_1}
\end{align*}$$

To achieve an initially plausible weight scheme, the following initializations will chosen: $w_0=-1$, $w_1=3$.  The learning rate will start as $\alpha=0.1$ and will decay by $1/((\textrm{number of proteins}) \cdot (\textrm{number of epochs}))$ after each update.  The Mash p-value threshold will be $\tau=1/(\textrm{number of proteins})\approx 9\cdot10^{-6}$.

### Model 3 (continuous, one variable)

Let $y$ be the true value for the continuous condition we are interested in (such as the concentration of sodium chloride).  We want to create a probability density function $f(\hat{y})$ that models the probability of the true crystallization condition being equal to some potential $\hat{y}$.  We want to maximize the probability assigned to some small interval around the value of the true condition, $\int_{y(1-\delta)}^{y(1+\delta)}f(\hat{y})d\hat{y}$ for some small $\delta$, or, equivalently, we want to minimize the area of the fit density function that falls outside of that interval, $1-\int_{y(1-\delta)}^{y(1+\delta)}f(\hat{y})d\hat{y}$.  Because many of the conditions are right skewed (and non-zero since the zero/non-zero case is dealt with by the binary model), we choose an interval with bounds multiplicatively rather than additively dependent on $y$.  The two exceptions are that pH and temperature will depend additively on $\delta$ since they are non-skewed and can be zero, but the model is otherwise equivalent.  This probability density can be created by applying a Gaussian kernel to a set of known crystallization conditions $\mathbf{x}$ from similar proteins.  With $x_i$ as the condition of the crystallization condition for protein $i$, $p_i$ as the Mash p-value of the similarity between protein $i$ and the target protein, $h_i$ as the bandwidth of the kernel element for protein $i$, $n_p$ as the total number of proteins with Mash p-values less than $\tau$, $\bar{x}$ as the average of the crystallization conditions of all proteins excluding the protein of interest, and $\eta$ as the standard deviation of the crystallization conditions of all proteins excluding the protein of interest, this density function can be written as follows.

$$\begin{equation}
f(\hat{y})=\frac{1}{n_p+1}\frac{1}{\sqrt{2\pi}}\cdot\frac{1}{\eta}\exp\left[{-\frac{\left(\frac{\hat{y}-\bar{x}}{\eta}\right)^2}{2}}\right] + \frac{n_p}{n_p+1}\cdot\frac{1}{n_p\sqrt{2\pi}}\sum_{i}\frac{1}{h_i}\exp\left[{-\frac{\left(\frac{\hat{y}-x_i}{h_i}\right)^2}{2}}\right] \textrm{ for } i \textrm{ such that } p_i<\tau
\end{equation}$$

Intuitively, this is a kernel density estimate combining the distribution of crystallization conditions for all proteins and the distribution of crystallization conditions for only proteins similar to the protein of interest.  However, two issues arise: not all of these similar proteins are equally similar, so they should not be weighted equally, and the optimal bandwidth $h_i$ of each term is unknown.  Both of these issues can be solved simultaneously by allowing $h_i$ to be a function of $s_i$, the sequence identity (specifically, 1 minus the Mash distance between protein $i$ and the protein of interest).  This function should be continuous and likely decreasing on $[0,1]$ because more similar proteins likely should have a smaller bandwidth for their kernels, and the function should have a codomain of $(0,\infty)$ because all the weights should be positive, but some could be much larger than others.  Therefore, the relationship between $h_i$ and $s_i$ will be given by 

$$\begin{equation}
h_i=\frac{c\sigma(w_1s_i+w_0)}{\int_0^1\sigma(w_1x+w_0)dx}=\frac{c\sigma(w_1s_i+w_0)}{\frac{\ln{(e^{-w_1}+e^{w_0})}-\ln(1+e^{w_0})}{w_1}+1}
\end{equation}$$

 where $\sigma$ is the sigmoid function and $c$ is a scaling value.  Letting $\mathbf{x}$ be the vector of conditions of the similar proteins and $\mathbf{s}$ be the vector of sequence identities of the similar proteins, define the loss as 
 
 $$L(y, \mathbf{x}, \mathbf{s}, \bar{x}, \eta, \delta, \beta)=1-\int_{y(1-\delta)}^{y(1+\delta)}f(\hat{y})d\hat{y}+\beta(\eta-c)^2$$
 
 with $f$ and $h_i$ as defined above.  We choose to regularize $c$ against $\eta$ because a naive bandwidth should be about the standard deviation of the whole observed condition range, not 0.  In practice, letting $c$ vary often causes the model to break down because values of $x_i$ close to $y$ generate an extremely steep gradient for $c$, pushing $c$ very close to 0.  By creating extremely sharp peaks of density at each $x_i$, this undermines the effort to create a smooth probability density and makes numerical integration essentially impossible.  Thus, we will fix $c$ at $\eta$.   While this forces the average bandwidth to be the standard deviation of all values for the condition, the function is capable of becoming much larger near 0 than near 1, and protein similarities near 0 often do not pass the p-value threshold for inclusion.  Thus, in practice, bandwidths can become as small as necessary even with a fixed $c$.  Still, for generality, we will treat $c$ as a variable.

 (In the implementation of this model, all values $y$ and $x$ of the crystallization condition will be divided by their condition means to account for the considerable differences in scale between conditions while maintaining the general right skewedness and positive values of all conditions.  When predicting values or ranges on the original scale, we can simply generate an estimate on this altered scale and multiply by the condition's mean.)

The specified model enables the fitting of three parameters: $w_0$, $w_1$, and $c$.  Let $h_i$ be as described above.  Let $\sigma_i=\sigma(w_1s_i+w_0)$.  Let $U$ be the $\sigma$ normalization term $\int_0^1\sigma(w_1x+w_0)dx=\frac{\ln{(e^{-w_1}+e^{w_0})}-\ln(1+e^{w_0})}{w_1}+1$.  Let $d_i=y-x_i$.  Let $z_i=\left(\frac{d_i}{h_i}\right)$.  Let $m = e^{w_0}+e^{-w_1}$.  Applying the chain rule, the sum rule for derivatives, and the fact that for $\sigma(x)$, $\frac{d\sigma}{dx}=\sigma(x)(1-\sigma(x))$, we obtain the following:

$$\begin{align*} 
\frac{\partial f(\hat{y})}{\partial w_0} &= \sum_i\frac{\partial f(\hat{y})}{\partial h_i}\frac{\partial h_i}{\partial w_0}\\
\frac{\partial f(\hat{y})}{\partial w_1} &= \sum_i\frac{\partial f(\hat{y})}{\partial h_i}\frac{\partial h_i}{\partial w_1}\\
\frac{\partial f(\hat{y})}{\partial c} &= \sum_i\frac{\partial f(\hat{y})}{\partial h_i}\frac{\partial h_i}{\partial c}\\
\frac{\partial f(\hat{y})}{\partial h_i} &= \frac{1}{(n_p+1)\sqrt{2\pi}}\frac{\exp({-z_i^2/2})(z_i^2-1)}{h_i^2}\\
\frac{\partial h_i}{\partial w_0} &= \frac{c\sigma_i(1-\sigma_i)}{U}-\frac{c\sigma_i(\frac{e^{w_0}}{m}-\frac{e^{w_0}}{1+e^{w_0}})}{w_1U^2}\\
\frac{\partial h_i}{\partial w_1} &= \frac{s_ic\sigma_i(1-\sigma_i)}{U}-\frac{c\sigma_i[\frac{-w_1e^{-w_1}}{e^{-w_1}+e^{w_0}} - (\ln(e^{-w_1}+e^{w_0})-\ln(1+e^{w_0}))]}{w_1^2U^2}\\
\frac{\partial h_i}{\partial c} &= \frac{h_i}{c}\\
\end{align*}$$

However, we are actually interested in the integrals of these quantities, so applying the Leibniz integral rule gives the following:

$$\begin{equation}
\begin{aligned} 
\frac{\partial L(y, \mathbf{x}, \mathbf{s}, \bar{x}, \eta, \delta, \beta)}{\partial w_0} &= -\int_{y(1-\delta)}^{y(1+\delta)} \frac{\partial f(\hat{y})}{\partial w_0}d\hat{y}\\   
\frac{\partial L(y, \mathbf{x}, \mathbf{s}, \bar{x}, \eta, \delta, \beta)}{\partial w_1} &=  -\int_{y(1-\delta)}^{y(1+\delta)} \frac{\partial f(\hat{y})}{\partial w_1}d\hat{y}\\ 
\frac{\partial L(y, \mathbf{x}, \mathbf{s}, \bar{x}, \eta, \delta, \beta)}{\partial c} &=  -\int_{y(1-\delta)}^{y(1+\delta)} \frac{\partial f(\hat{y})}{\partial c}d\hat{y}+2\beta (c-\eta)\\
\end{aligned}
\end{equation}$$

Because of the memory requirements involved in manipulating all the amino acid identity scores at once, we will use stochastic gradient descent to pick a protein at random, determine its amino acid identity against all the other proteins, compute its density function, and update the weights according to the loss.  With a learning rate $\alpha$, the update statements will be as follows:

$$\begin{equation}
\begin{aligned}
w_0&\leftarrow w_0-\alpha\frac{\partial L(y, \mathbf{x}, \mathbf{s}, \bar{x}, \eta, \delta, \beta)}{\partial w_0}\\
w_1&\leftarrow w_1-\alpha\frac{\partial L(y, \mathbf{x}, \mathbf{s}, \bar{x}, \eta, \delta, \beta)}{\partial w_1}\\
c&\leftarrow c-\alpha\frac{\partial L(y, \mathbf{x}, \mathbf{s}, \bar{x}, \eta, \delta, \beta)}{\partial c}
\end{aligned}
\end{equation}$$

The partial derivative of the density function with respect to each parameter will be computed exactly using equation 3, but the Leibniz integrals in equation 4 will be approximated with a left Riemann sum and a $\Delta\hat{y}$ of $y/100$.

By linearity of expectation, the expectation of $f$ is simply $$\frac{1}{n_p+1}\bar{x}+\frac{n_p}{n_p+1}\frac{1}{n_p}\sum\limits_{i}x_i$$

Because we do not need extreme precision, the approximate mode of the distribution can be found by evaluating the PDF from $\min(\bar{x},\min(\{x_i|x_i\in\mathbf{x}\}))$ to $\max(\bar{x},\max(\{x_i|x_i\in\mathbf{x}\}))$ with a step size of the difference divided by 1,000 and recording the value of the condition at the maximum value of the PDF.  Because the density of an individual kernel input decreases on either side of its mean, the mode is guaranteed to be between these bounds.  Because numeric integration over a large sum of variables is computationally costly and we do not need extreme precision, we can find an estimated 95\% confidence interval from the 2.5<sup>th</sup> percentile to the 97.5<sup>th</sup> percentile of the kernel density by iterating over crystallization condition values from either end of the distribution.  For the 2.5<sup>th</sup> percentile, we begin iterating upwards from $\hat{y}=\max(\min(\textrm{condition value for all proteins}),\min(\Phi^{-1}(0.025)\eta+\bar{x}, \{\min(\Phi^{-1}(0.025)h_i+x_i)|x_i\in\mathbf{x}\}))$ and taking steps of the same size as for the mode until 

$$\frac{1}{n_p+1}\Phi\left(\frac{\hat{y}-\bar{x}}{\eta}\right)+\frac{n_p}{n_p+1}\cdot\frac{1}{n_p}\sum_{i}\Phi\left(\frac{\hat{y}-x_i}{h_i}\right) \geq 0.025$$

We take this $\hat{y}$ as the 2.5<sup>th</sup> percentile.  Likewise, for the 97.5<sup>th</sup> percentile, we begin iterating downwards from $\hat{y}=\min(\max(\textrm{condition value for all proteins}),\max(\Phi^{-1}(0.975)\eta+\bar{x},\max(\{\Phi^{-1}(0.975)h_i+x_i)|x_i\in\mathbf{x}\}))$ and taking steps of the same size until

$$\frac{1}{n_p+1}\Phi\left(\frac{\hat{y}-\bar{x}}{\eta}\right)+\frac{n_p}{n_p+1}\cdot\frac{1}{n_p}\sum_{i}\Phi\left(\frac{\hat{y}-x_i}{h_i}\right) < 0.975$$

We then take the $\hat{y}$ before the current one (the last one where the expression was greater than 0.975) as the 97.5<sup>th</sup> percentile.  As a proof sketch that the minimum of any term's 2.5<sup>th</sup> percentile is less than or equal to the 2.5<sup>th</sup> percentile of the whole kernel density, consider that for any other term besides this minimizer, that term's 2.5<sup>th</sup> percentile must be larger by construction, so it contributes more density to the kernel above its own 2.5<sup>th</sup> percentile and therefore above the minimum 2.5<sup>th</sup> percentile than below either.  Therefore, this other term shifts the density of the total kernel density upwards, adding more weight above the minimum 2.5<sup>th</sup> percentile, guaranteeing that the overall 2.5<sup>th</sup> percentile is larger than the minimum of the individual terms' 2.5<sup>th</sup> percentiles.  The proof for the 97.5<sup>th</sup> percentile is analogous. 
 Further bounding the search range by the minimum and maximum observed values of the condition ensures that a protein with only a few distantly related proteins doesn't require a massive search space due to very large bandwidths in the kernel density.

To achieve an initially plausible bandwidth scheme, the following initializations will chosen: $w_0=-1$, $w_1=-2, c=\eta$.  The following image shows the bandwidths produced by these parameters (with $\eta=1$) with more similar proteins having a lower bandwidth as expected.   

<p align="center">
<img src="images/continuous.png" width="300" height="200">
</p>

The learning rate will start as $\alpha=0.1$ and will decay by $1/((\textrm{number of proteins}) \cdot (\textrm{number of epochs}))$ after each update.  The Mash p-value threshold will be $\tau=1/(\textrm{number of proteins})\approx 9\cdot10^{-6}$.

## Data curation

## Results
