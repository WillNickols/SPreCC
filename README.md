# SPreCC
**S**imilarity-based **Pre**diction of **C**rystallization **C**onditions

## Overview

Knowledge of the three-dimensional structures of proteins is essential for understanding their biological functions and developing targeted molecular therapies.  The most common technique for elucidating the structure of a protein is X-ray crystallography, but obtaining high quality crystals for X-ray diffraction remains the bottleneck of this technique.  Predicting chemical conditions that would produce high quality crystals from the amino acid sequence of a protein would considerably speed this process.  Previous efforts have attempted to determine the optimal pH for crystallization from a protein's amino acid sequence and its pI or predict a list of chemical conditions for crystallization from its amino acid sequence.  However, to our knowledge, no attempt has been made to predict the technique, pH, temperature, chemical conditions, and concentrations of those chemical conditions all together from a protein's amino acid sequence.  Here, we present and evaluate SPreCC, a method of predicting categorical, continuous, and presence/absence information for crystalliation conditions based on amino acid sequence similarity.

## Table of Contents  
* [Introduction](#introduction)
* [Models](#models)
* [Data Curation](#data-curation)
* [Results](#results)
* [Authors and Acknowledgements](#authors-and-acknowledgements)

## Introduction

## Models

We will present four models for different types of crystallization condition data.  The first will be a model for predicting the presence/absence status of chemical conditions such as sodium chloride or tris.  The second will be a model for predicting categorical values, which will be applied only to predicting the crystallization technique to be used.  The third will be a model for predicting concentrations of chemical conditions like the molarity of sodium chloride or the percent volume per volume of dimethyl sulfoxide.  The fourth will be a model for simultaneously predicting concentrations and polymer lengths for chemical conditions like polyethylene glycol (PEG).  A single condition might use multiple models; for example, sodium chloride uses both the presence/absence model for predicting whether it should be in the crystallization mixture and the concentration model for predicting its optimal concentration if it is present.  In general, sequence similarities will be determined using [Mash](https://mash.readthedocs.io/en/latest/index.html), and protein indexing will be over proteins with the relevant data available from the [Protein Data Bank](https://www.rcsb.org/).

### Model 1 (presence/absence)

Let $y$ be the true binary value we are interested in (such as whether or not sodium chloride is present).  Let our prediction for the binary value be $\hat{y}$ such that 

$$\hat{y} = \frac{\bar{x}}{n_p+1} + \frac{n_p}{n_p+1}\cdot\frac{\sum\limits_{i} x_i\sigma(w_1s_i+w_0)}{\sum\limits_{i} \sigma(w_1s_i+w_0)} \textrm{ for } i \textrm{ such that } p_i<\tau$$

 where $x_i$ is the binary value for protein $i$ (e.g. whether sodium chloride was present for protein $i$), $s_i\in[0,1]$ is 1 minus the [Mash](https://mash.readthedocs.io/en/latest/index.html) distance between protein $i$ and the protein of interest, $\sigma$ is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function), $p_i$ is the Mash p-value of the similarity between protein $i$ and the target protein (how likely the two proteins are to have their reported degree of similarity by chance), $\bar{x}$ is the average value of the binary condition across all the dataset excluding the protein of interest, and $n_p$ is the total number of proteins with Mash p-values less than $\tau$.  Intuitively, we are taking a weighted average between the binary values from all the proteins and the binary values from related proteins.  This ensures that the model still gives an estimate for proteins with no similar proteins in the database while also allowing predictions for proteins with even a few similar proteins to be mostly determined by the conditions of those similar proteins.  Within the term corresponding to similar proteins, $\sigma(w_1s_i+w_0)$ is the weight for the crystallization condition of protein $i$, and the denominator normalizes the calculation.  Each weight should be some value between 0 and 1, and we expect greater sequence identities to correspond to heavier weights, but the model allows flexibility in how much some amount of additional sequence identity should increase the weight.  This weighting scheme allows much more flexibility and speed than, for example, incorporating the distance of every protein or ranking the most similar proteins.  It allows a variable number of inputs, preventing a need for as many independent weights as there are proteins, and it allows the weight to be determined directly from the sequence similarity rather than from some ranking of similarities.  We will attempt to minimize the negative log-likelihood loss: 
 $$L(y,\hat{y}, \beta)=-[y\ln(\hat{y}) + (1-y)\ln(1-\hat{y})] + \beta||\boldsymbol{w}||^2$$

The specified model enables the fitting of two parameters: $w_0$ and $w_1$.  Let $\sigma_i=\sigma(w_1s_i+w_0)$.  Applying the chain rule, we obtain the following:

$$\begin{align*} 
\frac{\partial L(\hat{y},y)}{\partial w_0} &= \frac{\partial L(\hat{y},y)}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_0} +2\beta w_0\\ 
&= -\left[\frac{y}{\hat{y}}-\frac{1-y}{1-\hat{y}}\right]\cdot \frac{\left(\sum\limits_{i}\sigma_i\right)\left(\sum\limits_{i}x_i\sigma_i(1-\sigma_i)\right)-\left(\sum\limits_{i}x_i\sigma_i\right)\left(\sum\limits_{i}\sigma_i(1-\sigma_i)\right)}{\left(\sum\limits_{i}\sigma_i\right)^2}+2\beta w_0\\
\frac{\partial L(\hat{y},y)}{\partial w_1} &=  \frac{\partial L(\hat{y},y)}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_1}+2\beta w_1\\ 
&=  -\left[\frac{y}{\hat{y}}-\frac{1-y}{1-\hat{y}}\right]\cdot \frac{\left(\sum\limits_{i}\sigma_i\right)\left(\sum\limits_{i}x_i\sigma_i(1-\sigma_i)s_i\right)-\left(\sum\limits_{i}x_i\sigma_i\right)\left(\sum\limits_{i}\sigma_i(1-\sigma_i)s_i\right)}{\left(\sum\limits_{i}\sigma_i\right)^2}+2\beta w_1
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
 
$$L(y,\hat{y}, \beta)=-\sum_{k=1}^K y_k \ln (\hat{y}_k) + \beta||\boldsymbol{w}||^2$$ 

where $K$ is the number of classes.

The specified model enables the fitting of two parameters: $w_0$ and $w_1$.  Because of the loss specification, the gradient with respect to $w_0$ or $w_1$ will only pass through the chain rule with the $\hat{y}_k$ that corresponds to the correct $y_k$.  Let $\sigma_i=\sigma(w_1s_i+w_0)$.  Applying the chain rule, we obtain the following:

$$\begin{align*} 
\frac{\partial L(\hat{y},y)}{\partial w_0} &= \frac{\partial L(\hat{y},y)}{\partial \hat{y_k}} \cdot \frac{\partial \hat{y_k}}{\partial w_0}+2\beta w_0\\ 
&= -\left[\frac{1}{\hat{y_k}}\right]\cdot \frac{\left(\sum\limits_{i}\sigma_i\right)\left(\sum\limits_{i}x_{k,i}\sigma_i(1-\sigma_i)\right)-\left(\sum\limits_{i}x_{k,i}\sigma_i\right)\left(\sum\limits_{i}\sigma_i(1-\sigma_i)\right)}{\left(\sum\limits_{i}\sigma_i\right)^2}+2\beta w_0\\
\frac{\partial L(\hat{y},y)}{\partial w_1} &=  \frac{\partial L(\hat{y},y)}{\partial \hat{y_k}} \cdot \frac{\partial \hat{y_k}}{\partial w_1}+2\beta w_1\\ 
&=  -\left[\frac{1}{\hat{y_k}}\right]\cdot \frac{\left(\sum\limits_{i}\sigma_i\right)\left(\sum\limits_{i}x_{k,i}\sigma_i(1-\sigma_i)s_i\right)-\left(\sum\limits_{i}x_{k,i}\sigma_i\right)\left(\sum\limits_{i}\sigma_i(1-\sigma_i)s_i\right)}{\left(\sum\limits_{i}\sigma_i\right)^2}+2\beta w_1
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
 
 $$L(y, \mathbf{x}, \mathbf{s}, \bar{x}, \eta, \delta, \beta)=1-\int_{y(1-\delta)}^{y(1+\delta)}f(\hat{y})d\hat{y}+\beta(\eta-c)^2 + \beta||\boldsymbol{w}||^2$$
 
 with $f$ and $h_i$ as defined above.  We choose to regularize $c$ against $\eta$ because a naive bandwidth should be about the standard deviation of the whole observed condition range, not 0.  In practice, letting $c$ vary often causes the model to break down because values of $x_i$ close to $y$ generate an extremely steep gradient for $c$, pushing $c$ very close to 0.  By creating extremely sharp peaks of density at each $x_i$, this undermines the effort to create a smooth probability density and makes numerical integration essentially impossible.  Thus, we will fix $c$ at $\eta$.   While this forces the average bandwidth to be the standard deviation of all values for the condition, the function is capable of becoming much larger near 0 than near 1, and protein similarities near 0 often do not pass the p-value threshold for inclusion.  Thus, in practice, bandwidths can become as small as necessary even with a fixed $c$.  Still, for generality, we will treat $c$ as a variable.

 (In the implementation of this model, all values $y$ and $x$ of the crystallization condition will be divided by their condition means to account for the considerable differences in scale between conditions while maintaining the general right skewedness and positive values of all conditions.  When predicting values or ranges on the original scale, we can simply generate an estimate on this altered scale and multiply by the condition's mean.)

The specified model enables the fitting of three parameters: $w_0$, $w_1$, and $c$.  Let $h_i$ be as described above.  Let $\sigma_i=\sigma(w_1s_i+w_0)$.  Let $U$ be the $\sigma$ normalization term $\int_0^1\sigma(w_1x+w_0)dx=\frac{\ln{(e^{-w_1}+e^{w_0})}-\ln(1+e^{w_0})}{w_1}+1$.  Let $d_i=y-x_i$.  Let $z_i=\left(\frac{d_i}{h_i}\right)$.  Let $m = e^{w_0}+e^{-w_1}$.  Applying the chain rule, we obtain the following:

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
\frac{\partial L(y, \mathbf{x}, \mathbf{s}, \bar{x}, \eta, \delta, \beta)}{\partial w_0} &= -\int_{y(1-\delta)}^{y(1+\delta)} \frac{\partial f(\hat{y})}{\partial w_0}d\hat{y}+2\beta w_0\\   
\frac{\partial L(y, \mathbf{x}, \mathbf{s}, \bar{x}, \eta, \delta, \beta)}{\partial w_1} &=  -\int_{y(1-\delta)}^{y(1+\delta)} \frac{\partial f(\hat{y})}{\partial w_1}d\hat{y}+2\beta w_1\\ 
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

### Model 4 (continuous, two variables)

Let $(y_1,y_2)$ be the true value for the two-dimensional continuous condition we are interested in (such as the concentration and length of PEG).  We want to create a probability density function $f(\hat{y_1},\hat{y_2})$ that models the probability of the true crystallization condition being equal to some potential $(\hat{y_1},\hat{y_2})$.  We want to maximize the probability assigned to some small interval around the value of the true condition, $\int_{y_1(1-\delta)}^{y_1(1+\delta)}\int_{y_2(1-\delta)}^{y_2(1+\delta)}f(\hat{y_1},\hat{y_2})d\hat{y_1}d\hat{y_2}$ for some small $\delta$, or, equivalently, we want to minimize the area of the fit density function that falls outside of that interval, $1-\int_{y_1(1-\delta)}^{y_1(1+\delta)}\int_{y_2(1-\delta)}^{y_2(1+\delta)}f(\hat{y_1},\hat{y_2})d\hat{y_1}d\hat{y_2}$.  This probability density can be created by applying a Gaussian kernel to a set of known crystallization conditions $\mathbf{x}$ from similar proteins that contained the condition of interest.  With $(x_{1,i}, x_{2,i})$ as the value of the crystallization condition for protein $i$, $p_i$ as the Mash p-value of the similarity between protein $i$ and the target protein, $(h_{1,i}, h_{2,i})$ as the bandwidths of the kernel element for protein $i$, $n_p$ as the total number of proteins with Mash p-values less than $\tau$, $(\bar{x_1}, \bar{x_2})$ as the average of the crystallization conditions of all proteins excluding the protein of interest, and $\eta$ as the standard deviation of the crystallization conditions of all proteins excluding the protein of interest, this density function can be written as follows.

$$\begin{align*} 
\begin{split}
f(\hat{y_1},\hat{y_2})&=\frac{1}{n_p+1}\frac{1}{2\pi}\cdot\frac{1}{\eta_1\eta_2}\exp\left[{-\frac{\left(\frac{\hat{y_1}-\bar{x_1}}{\eta_1}\right)^2+\left(\frac{\hat{y_2}-\bar{x_2}}{\eta_2}\right)^2}{2}}\right]\\ 
&+ \frac{1}{n_p+1}\cdot\frac{1}{2\pi}\sum_{i}\frac{1}{h_{1,i}h_{2,i}}\exp\left[{-\frac{\left(\frac{\hat{y_1}-x_{1,i}}{h_{1,i}}\right)^2+\left(\frac{\hat{y_2}-x_{2,i}}{h_{2,i}}\right)^2}{2}}\right] \textrm{ for } i \textrm{ such that } p_i<\tau
\end{split}
\end{align*}$$

Intuitively, this is a kernel density estimate weighing together the distribution of crystallization conditions for all proteins and the distribution of crystallization conditions for only proteins similar to the protein of interest.  However, two issues arise: not all of these similar proteins are equally similar, so they should not be weighted equally, and the optimal bandwidths $(h_{1,i}, h_{2,i})$ of each term are unknown.  Both of these issues can be solved simultaneously by allowing $(h_{1,i}, h_{2,i})$ to be a function of $s_i$, the sequence identity (specifically, 1 minus the Mash distance between protein $i$ and the protein of interest).  This function should be continuous and decreasing on $[0,1]$ because more similar proteins should have a smaller bandwidth for their kernels, and the function should have a codomain of $(0,\infty)$ because all the weights should be positive, but some could be much larger than others.  Therefore, the relationship between $(h_{1,i}, h_{2,i})$ and $s_i$ will be given by 

$$\begin{equation}
h_{j,i}=\frac{c_j\sigma(w_{j,1}s_i+w_{j,0})}{\int_0^1\sigma(w_{j,1}x+w_{j,0})dx}=\frac{c_j\sigma(w_{j,1}s_i+w_{j,0})}{\frac{\ln{(e^{-w_{j,1}}+e^{w_{j,0}})}-\ln(1+e^{w_{j,0}})}{w_{j,1}}+1}
\end{equation}$$

 for $j\in\{1,2\}$ where $\sigma$ is the sigmoid function and $c_j$ is a scaling value.  Letting $\mathbf{x}$ be the vector of conditions of the similar proteins and $\mathbf{s}$ be the vector of sequence identities of the similar proteins, define the loss as 
 
 $$L((y_1,y_2), \mathbf{x}, \mathbf{s}, (\bar{x_1}, \bar{x_2}), (\eta_1,\eta_2), (\delta_1,\delta_2), \beta)=1-\int_{y_1(1-\delta)}^{y_1(1+\delta)}\int_{y_2(1-\delta)}^{y_2(1+\delta)}f(\hat{y_1},\hat{y_2})d\hat{y_1}d\hat{y_2}+\beta||\boldsymbol{\eta}-\mathbf{c}||^2 + \beta||\boldsymbol{w}||^2$$
 
 with $f$ and $(h_{1,i}, h_{2,i})$ as defined above.  We choose to regularize $\mathbf{c}$ against $\boldsymbol{\eta}$ because a naive bandwidth should be about the standard deviation of the whole observed condition range, not 0.  In practice, letting $c_1$ and $c_2$ vary often causes the model to break down because values of $x_i$ close to $y$ generate an extremely steep gradient for $c_1$ and $c_2$, pushing them very close to 0.  By creating extremely sharp peaks of density at each $x_i$, this undermines the effort to create a smooth probability density and makes numerical integration essentially impossible.  Thus, we will fix $c_1$ and $c_2$ at $\eta_1$ and $\eta_2$ respectively.  While fixing these values forces the average bandwidth to be the standard deviation of all values for the condition, the function is capable of becoming much larger near 0 than near 1, and protein similarities near 0 often do not pass the p-value threshold for inclusion.  Thus, in practice, bandwidths can become as small as necessary even with fixed $c_1$ and $c_2$.  Still, for generality, we will treat both as variables.

 (In the implementation of this model, all values $y$ and $x$ of the crystallization condition will be divided elementwise by their condition means to account for the considerable differences in scale between conditions while maintaining the general right skewedness and positive values of all conditions.  When predicting values or ranges on the original scale, we can simply generate an estimate on this altered scale and multiply by the condition's mean.)

The specified model enables the fitting of six parameters: $w_{1,0}$, $w_{1,1}$, $c_1$, $w_{2,0}$, $w_{2,1}$, and $c_2$.  Let $(h_{1,i}, h_{2,i})$ be as described above.  For $j\in\{1,2\}$, let $\sigma_{j,i}=\sigma(w_{j,1}s_i+w_{j,0})$.  Let $U_j$ be the $\sigma_j$ normalization term $\int_0^1\sigma(w_{j,1}x+w_{j,0})dx=\frac{\ln{(e^{-w_{j,1}}+e^{w_{j,0}})}-\ln(1+e^{w_{j,0}})}{w_{j,1}}+1$.  Let $d_{j,i}=y_j-x_{j,i}$.  Let $z_{j,i}=\left(\frac{d_{j,i}}{h_{j,i}}\right)$.  Let $m_j = e^{w_{j,0}}+e^{-w_{j,1}}$, $r_j=\frac{e^{w_{j,0}}}{m_j}-\frac{e^{w_{j,0}}}{1+e^{w_{j,0}}}$, $v_j=[\frac{-w_{j,1}e^{-w_{j,1}}}{m_j} - (\ln(m_j)-\ln(1+e^{w_{j,0}}))]/w_{j,1}$.  Applying the chain rule, we obtain the following:

$$\begin{align*}
\frac{\partial f(\hat{y_1},\hat{y_2})}{\partial w_{j,0}} &= \sum_i\frac{\partial f(\hat{y_1},\hat{y_2})}{\partial h_{j,i}}\frac{\partial h_{j,i}}{\partial w_{j,0}}\\
\frac{\partial f(\hat{y_1},\hat{y_2})}{\partial w_{j,1}} &= \sum_i\frac{\partial f(\hat{y_1},\hat{y_2})}{\partial h_{j,i}}\frac{\partial h_{j,i}}{\partial w_{j,1}}\\   
\frac{\partial f(\hat{y_1},\hat{y_2})}{\partial c_j} &= \sum_i\frac{\partial f(\hat{y_1},\hat{y_2})}{\partial h_{j,i}}\frac{\partial h_{j,i}}{\partial c_j}\\   
\frac{\partial f(\hat{y_1},\hat{y_2})}{\partial h_{1,i}} &= \frac{1}{(n_p+1)2\pi}\frac{\exp({-z_{2,i}^2/2})}{h_{2,i}}\frac{\exp({-z_{1,i}^2/2})(z_{1,i}^2-1)}{h_{1,i}^2}\\ 
\frac{\partial f(\hat{y_1},\hat{y_2})}{\partial h_{2,i}} &= \frac{1}{(n_p+1)2\pi}\frac{\exp({-z_{1,i}^2/2})}{h_{1,i}}\frac{\exp({-z_{2,i}^2/2})(z_{2,i}^2-1)}{h_{2,i}^2}\\
\partial h_{j,i}/\partial w_{j,0} &= [c_j\sigma_{j,i}(1-\sigma_{j,i})]/U_j-[c_j\sigma_{j,i}r_j]/[w_{j,1}(U_j)^2]\\
\partial h_{j,i}/\partial w_{j,1} &= [s_i\cdot c_j\sigma_{j,i}(1-\sigma_{j,i})]/U_j-[c_j\sigma_{j,i}v_j]/[w_{j,1}(U_j)^2]\\
\frac{\partial h_{j,i}}{\partial c_j} &= \frac{h_{j,i}}{c_j}\\
\end{align*}$$

However, we are actually interested in the integrals of these quantities, so applying the Leibniz integral rule gives the following:

$$\begin{equation}
\begin{aligned} 
\frac{\partial L((y_1,y_2), \mathbf{x}, \mathbf{s}, (\bar{x_1}, \bar{x_2}), (\eta_1,\eta_2), (\delta_1,\delta_2), \beta)}{\partial w_{j,0}} &= -\int_{y_1(1-\delta)}^{y_1(1+\delta)}\int_{y_2(1-\delta)}^{y_2(1+\delta)}\frac{f(\hat{y_1},\hat{y_2})}{\partial w_{j,0}}d\hat{y_1}d\hat{y_2}+2\beta w_{j,0}\\   
\frac{\partial L((y_1,y_2), \mathbf{x}, \mathbf{s}, (\bar{x_1}, \bar{x_2}), (\eta_1,\eta_2), (\delta_1,\delta_2), \beta)}{\partial w_{j,1}} &=  -\int_{y_1(1-\delta)}^{y_1(1+\delta)}\int_{y_2(1-\delta)}^{y_2(1+\delta)}\frac{f(\hat{y_1},\hat{y_2})}{\partial w_{j,1}}d\hat{y_1}d\hat{y_2}+2\beta w_{j,1}\\ 
\frac{\partial L((y_1,y_2), \mathbf{x}, \mathbf{s}, (\bar{x_1}, \bar{x_2}), (\eta_1,\eta_2), (\delta_1,\delta_2), \beta)}{\partial c_j} &=  -\int_{y_1(1-\delta)}^{y_1(1+\delta)}\int_{y_2(1-\delta)}^{y_2(1+\delta)}\frac{f(\hat{y_1},\hat{y_2})}{\partial c_{j}}d\hat{y_1}d\hat{y_2}+2\beta (c_j-\eta_j)\\
\end{aligned}
\end{equation}$$

Because of the memory requirements involved in manipulating all the amino acid identity scores at once, we will use stochastic gradient descent to pick a protein at random, determine its amino acid identity against all the other proteins, compute its density function, and update the weights according to the loss.  With a learning rate $\alpha$, the update statements will be as follows:

$$\begin{equation}
\begin{aligned}
w_{j,0}&\leftarrow w_{j,0}-\alpha\frac{\partial L((y_1,y_2), \mathbf{x}, \mathbf{s}, (\bar{x_1}, \bar{x_2}), (\eta_1,\eta_2), (\delta_1,\delta_2), \beta)}{\partial w_{j,0}}\\
w_{j,1}&\leftarrow w_{j,1}-\alpha\frac{\partial L((y_1,y_2), \mathbf{x}, \mathbf{s}, (\bar{x_1}, \bar{x_2}), (\eta_1,\eta_2), (\delta_1,\delta_2), \beta)}{\partial w_{j,1}}\\
c_j&\leftarrow c_j-\alpha\frac{\partial L((y_1,y_2), \mathbf{x}, \mathbf{s}, (\bar{x_1}, \bar{x_2}), (\eta_1,\eta_2), (\delta_1,\delta_2), \beta)}{\partial c_j}
\end{aligned}
\end{equation}$$

The partial derivative of the density function with respect to each parameter will be computed exactly using equation 3, but the Leibniz integrals in equation 4 will be approximated with a left Riemann sum and a $\Delta\hat{y_j}$ of $y_j/100$.  

By linearity of expectation, the expectation of $f$ is simply $$\frac{1}{n_p+1}\bar{x}+\frac{n_p}{n_p+1}\frac{1}{n_p}\sum\limits_{i}x_{i}$$

Because we do not need extreme precision, the approximate mode of the distribution can be found by evaluating the PDF from $\min(\bar{x},\min(\{x_{i}|x_{i}\in\mathbf{x}\}))$ to $\max(\bar{x},\max(\{x_{i}|x_{i}\in\mathbf{x}\}))$ with a step size of the difference divided by 1,000 and recording the crystallization condition at the largest value of the PDF.  Here, min indicates the element-wise minimum of the two-element vector of conditions (i.e. the minimum of $x_{j,i}$ over all $i$ for $j=1$ and $j=2$ separately).  Because the density of an individual kernel input decreases on all sides of its mean, the mode is guaranteed to be between these bounds. 
 Because numeric integration over a large sum of variables is computationally costly and we do not need extreme precision, we can use the marginal density for each of the two elements of the condition to find an estimated 95\% confidence interval from the 2.5<sup>th</sup> percentile to the 97.5<sup>th</sup> percentile of the kernel density by iterating over crystallization condition values from either end of their marginal distributions.  For the 2.5<sup>th</sup> percentile, we begin iterating upwards from $\hat{y_j}=\max(\min(\textrm{condition value for all proteins}),\min(\Phi^{-1}(0.025)\eta+\bar{x_j}, \{\min(\Phi^{-1}(0.025)h_{j,i}+x_{j,i})|x_{j,i}\in\mathbf{x}\}))$ and taking steps of the same size as for the mode until 

$$\frac{1}{n_p+1}\Phi\left(\frac{\hat{y_j}-\bar{x_j}}{\eta_j}\right)+\frac{n_p}{n_p+1}\cdot\frac{1}{n_p}\sum_{i}\Phi\left(\frac{\hat{y_j}-x_{j,i}}{h_{j,i}}\right) \geq 0.025$$
We take that $\hat{y_j}$ as the 2.5<sup>th</sup> percentile.  Likewise, for the 97.5<sup>th</sup> percentile, we begin iterating downwards from $\hat{y_j}=\min(\max(\textrm{condition value for all proteins}),\max(\Phi^{-1}(0.975)\eta+\bar{x_j},\max(\{\Phi^{-1}(0.975)h_{j,i}+x_{j,i})|x_{j,i}\in\mathbf{x}\}))$ and taking steps of the same size until

$$\frac{1}{n_p+1}\Phi\left(\frac{\hat{y_j}-\bar{x_j}}{\eta_j}\right)+\frac{n_p}{n_p+1}\cdot\frac{1}{n_p}\sum_{i}\Phi\left(\frac{\hat{y_j}-x_{j,i}}{h_{j,i}}\right) < 0.975$$

We then take the $\hat{y_j}$ before the current one (the last one where the expression was greater than 0.975) as the 97.5<sup>th</sup> percentile.  The proof sketch that the minimum of any term's 2.5<sup>th</sup> percentile is less than or equal to the 2.5<sup>th</sup> percentile of the marginal kernel density is analogous to the sketch given in the one-parameter system description.
 Further bounding the search range by the minimum and maximum observed values of the condition ensures that a protein with only a few distantly related proteins doesn't require a massive search space due to very large bandwidths in the kernel density.

To achieve an initially plausible bandwidth scheme, the following initializations will chosen for $j\in\{1,2\}$: $w_{j,0}=-1$, $w_{j,1}=-2, c_j=\eta_j$.  The learning rate will start as $\alpha=0.1$ and will decay by $1/((\textrm{number of proteins}) \cdot (\textrm{number of epochs}))$ after each update.  The Mash p-value threshold will be $\tau=1/(\textrm{number of proteins})\approx 9\cdot10^{-6}$.

## Data curation

Crystallization condition data was downloaded from the [Protein Data Bank (PDB)](https://www.rcsb.org/) on July 26<sup>th</sup> 2022 using [PyPDB](https://academic.oup.com/bioinformatics/article/32/1/159/1743800), and any proteins with crystallization data available were downloaded in their CIF file format from the PDB, resulting in 160,136 initial proteins.  Because many CIF files contain multiple amino acid chains (from proteins crystalized in dimers or complexes), we extracted the longest amino acid chain from each file, and we deduplicated any pairs of sequences and metadata free text that matched exactly.  We found that there were often dozens of proteins deposited with identical sequences and free text crystallization descriptions, usually from papers crystallizing a protein multiple times to evaluate different binding interactions.  This deduplication step removed 27,154 proteins (17% of the original proteins) for a final set of 132,982 proteins.  To obtain a realistic evaluation metric and further reduce the chance of train/evaluation overlap, we divided the proteins by date into train (before 2018, n=100,690), validation (2018-2019, n=16,733), and test (2020-2022, n=15,559) sets.  The fact that so many identical proteins had identical crystallization information and that (as the evaluation will show) proteins of a similar time are more informative of each other suggests that [previous evaluations](https://mlcb.github.io/mlcb2019_proceedings/papers/paper_3.pdf) that divided all proteins randomly without any apparent deduplication likely overestimated the effectivenss of their methods.

Because the crystallization conditions are provided in a free text field, we further processed the conditions to obtain standardized, numerical values.  Of the 116,713 proteins with a crystallization method provided, we grouped 116,529 (99.8%) of them into one of thirteen standardized categories with the remainder falling into an "other" category.  Initially, 106,220 proteins provided a designated pH value, and we were able to extract another 7,306 values from the free text and pH range fields for a total of 113,526 pH values.  117,068 entries provided the temperature at crystallization.  Of the 132,982 proteins with any crystallization information, 115,254 had some information in their free text fields, and 103,762 of those had information that included numerical values and recognizable units.  Because many entries listed the same chemicals with units of "percent," "percent volume per volume," "percent volume per weight," and "percent weight per volume," we combined these units as "percent" since the solvents usually had densities near that of water, so the different versions of "percent" were similar.  If the experimental setup was recorded in the [Biological Macromolecule Crystallization Database (BMCD)](http://bmcd.ibbr.umd.edu/), we replaced the free text field with the BMCD's parsed version because their parsing scripts show a higher level of reliability than ours.  Otherwise, we parsed the free text field with our own scripts to produce a list of chemicals and concentrations for each entry.  From these 103,762 entries, we determined the 100 most common chemicals and the concentration of each of these chemicals in each crystallization solution.  Chemicals beyond these top 100 were present in 168 (0.16% of usable entries) or fewer entries.  Of the usable entries, 67.3% (69,823) contained only chemicals in this top-100 list with the remainder containing rare or sufficiently misspelled chemicals.  Because most chemicals in most of the remaining entries (63.6%) were still in the top-100 list, we retained these entries for training and evaluation despite the model's inability to predict those conditions.  Finally, to ensure that mistyped concentrations did not skew our results, we excluded any instances of a chemical's concentration falling significantly outside the typical range.  Specifically, excluding values beyond three standard deviations of the original mean and then additionally excluding values outside of twenty standard deviations of this reduced set seemed to have the desired effect.  (This can be verified by examining the distribution of each chemical in the [figures folder](Figures/metadata).)

In general, to keep as many proteins for training and evaluation as possible, we included in training and evaluation any fields with recognizable and plausible information but excluded any fields without information.  For example, if a PDB entry only provided the protein's method of crystallization and pH at crystallization, we included that protein for training or evaluation on method and pH but excluded it from training or evaluation on temperature, chemical presence or absence, and chemical concentration.

## Results

The optimal crystallization condition prediction tool would be one which (1) accurately and with with high precision and recall predicts which chemicals are necessary for crystallization, (2) with a small margin of error predicts the concentration or a range of concentrations for those chemicals, (3) with a small margin of error predicts the value of or a range for the pH and temperature for crystallization, and (4) accurately predicts the optimal crystallization method to use.  We consider this last prediction the least useful because almost all proteins use vapor diffusion and those that do not (e.g. many membrane proteins) are well characterized enough before X-ray crystallography to know that an alterantive method will work better.  However, in the spirit of making our results comparable with previous work, we will include this prediction.

### Model evaluation
n_p, weights

### Presence/absence evaluation

### Concentration evaluation

## Authors and Acknowledgements
This project was envisioned, planned, and implemented by Will Nickols, Ben Tang, Jorge Guerra, Srihari Ganesh, and Andrew Lu.  The computations were run in part on the FASRC Cannon cluster supported by the FAS Division of Science Research Computing Group at Harvard University.  We would additionally like to thank the Curtis Huttenhower lab and the Long Nguyen lab at the Harvard School of Public Health for their computing resources.  The content and any mistakes herein are solely the responsibility of W. Nickols, B. Tang, J. Guerra, S. Ganesh, and A. Lu and do not reflect the views of the Huttenhower or Nguyen labs.
