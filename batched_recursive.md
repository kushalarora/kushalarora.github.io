---
layout: page
title: A Batched Recursive Neural Network.
---

General Formulation
===================

![Composition tree for sentence ’My dog like eating Pizza.’[]{data-label="fig:w_12345_comp"}](MSThesis/images/my_dog_likes_eating_pizza_comp){width="60.00000%"}

Let $D$ be a dataset containing tokenized sentences. Let $W=w_{1}w_{2}..w_{n}$ be a sentence of length $n$ in this data set. Let $t(W)$ be the tree structured assocated with the sentence $W$. Figure \[fig:w\_12345\_comp\] dipicts one such tree for a sentence $W_{12345}$.

Let $R(t,W)$ be the set of binary rules and leaf nodes used in the derivation of sentence $W$. Equation \[eq:r\_t\_w12345\] depicts the rule set $R(t,W_{12345})$ for the sentence $W_{12345}$ shown in Figure \[fig:w\_12345\_comp\].

$$\begin{aligned}
R(t,W_{12345})=\{ & w_{12}\leftarrow w_{1}w_{2},\nonumber \\
 & w_{34}\leftarrow w_{3}w_{4},\nonumber \\
 & w_{345}\leftarrow w_{3}w_{45},\nonumber \\
 & w_{12345}\leftarrow w_{12}w_{345},\nonumber \\
 & w_{1},w_{2},w_{3},w_{4},w_{5}\}\label{eq:r_t_w12345}\end{aligned}$$

Recurrent Neural Network Scoring Equation
-----------------------------------------

Let $d$ be the dimensions of the embedding space and $X_{i..k}$, $X_{k+1..j}$ be the $d\times1$ embedding vector corresponding to sequences $w_{i..k}$ and $w_{k+1..j}$ respectively. Using standard recursive neural network composition function, $X_{i..j}$ can be computed as $$X_{i..j}=f\left(H\left[\begin{array}{c}
X_{i..k}\\
X_{k+1..j}
\end{array}\right]\right)$$

$f$ here is a non linearity like $tanh$ and $H$ is a $d\times2d$ composition matrix.

The score for the node $X_{i..j}$ can calculated as $$s(w_{i..j}\leftarrow w_{i..k}w_{k+1..j})=s(w_{i..j})=g(UX_{i..j})$$

For example, let’s consider sentiment classification task. Let $K$ be the caridinality of the target classes, $T(w_{i..j})$ be a one-hot row vector with sentiment label for node $w_{i..j}$, then $s(w_{i..j})$ would be $$s(w_{i..j})=T(w_{i..j})lg(softmax(UX_{i..j}))$$

$U$ in this case is a $K\times d$ sentiment classification matrix.

The score of whole sentence $W$ can now be calculated as $$s(W;t)=\sum_{r\in R(t,W)}s(r)$$

Multiplying by -1 and exponentiating both sides,

$$exp(-s(W,t))=\prod_{r\in R(t,W)}exp(-s(r))\label{eq:rnn_s_W_form}$$

An Alternate formulation
------------------------

Let $\theta(W)$[^1] be a $n\times n\times n$ sparse tensor such that $$\theta(W)[i,j,k]=\begin{cases}
1 & w_{i...j}\leftarrow w_{i..k}w_{k+1..j}\in R(t,W)\\
0 & otherwise
\end{cases}\label{eq:theta_def}$$

Let $\pi(w_{i..j})$ be the inside score of the sequence $w_{i..j}$. The inside score is defined recursively as:

$$\pi(w_{ii})=exp(-s(w_{ii}))$$

and

$$\pi(w_{i..j};\theta)=\sum_{k=i+1}^{j-1}\theta(W)[i,j,k]exp(-s(w_{i..j}))\pi(w_{i..k})\pi(w_{k+1..j})\label{eq:pi_w_i_j}$$

As $\theta(W,t)$ is an indicator tensor, we can write score of the sentence $W$, $s(W,t)$ in terms of $\pi$ as:

$$exp(-s(W,t))=\pi(w_{1..n};\theta)$$

Hence,

$$s(W,t)=-lg(\pi(w_{1..n};\theta))$$

A Matrix Formulation for Inside Score.
--------------------------------------

Now, let $\Pi(W)$[^2] be a $n\times n$ matrix of inside scores whose rows are starting index and colums are the span sizes i.e. index $[i,j]$ would correspond to inside score $\pi(w_{i..(i+j)})$. First row of this matrix would be inside scores of all the leaf nodes $\pi(w_{ii})$ and each row would contain $n-i$ entries. Equation \[eq:Pi\_W\_12345\] shows an example matrix for $\Pi(W_{12345})$.

$$\Pi(W_{12345})=\left[\begin{array}{ccccc}
\pi(w_{1}) & \pi(w_{12}) & \pi(w_{123}) & \pi(w_{1234}) & \pi(w_{12345})\\
\pi(w_{2}) & \pi(w_{23}) & \pi(w_{234}) & \pi(w_{2345}) & 0\\
\pi(w_{3}) & \pi(w_{34}) & \pi(w_{345}) & 0 & 0\\
\pi(w_{4}) & \pi(w_{45}) & 0 & 0 & 0\\
\pi(w_{5}) & 0 & 0 & 0 & 0
\end{array}\right]\label{eq:Pi_W_12345}$$

Let $\Pi_{l}^{i}$ be a $l\times l$ submatrix starting at row $i+1$. $\bar{I}_{l}$ is the reflection of identity matrix $I_{l}$, $\Pi[i]_{l}$ is first $l$ elements of $i$th row of matrix $\Pi(W)$. Let $j=i+l$, $S_{ij}$ is a length $l$ row vector at element $i<k<j$ is $exp(-s(w_{i..j}\leftarrow w_{i..k}w_{k+1..j}))$.

$[i,j]$th index of matrix $\Pi$ can be calculated as follows $$\Pi[i,j]=\pi(w_{i..j})=(\theta[i,j,1:l]\circ S_{ij})(\bar{I}_{l}\circ\Pi_{l}^{i})\Pi[i]_{l}^{T}\label{eq:Pi_i_j_comp}$$

Here, $\circ$ is a elementwise multiplication.

Let’s try to compute $\pi(w_{12345})$ for sentence $W_{12345}$ from Figure $1$. In this case, $i=1$, $j=5$, and $l$ would be $j-i=4$. $\theta[i,j,1:l]$ and $S_{ij}$ in this case will be

$$\begin{aligned}
{1}
S_{15}= & [exp(-s(w_{12345}\leftarrow w_{1}w_{2345})),exp(-s(w_{12345}\leftarrow w_{12}w_{345})),\backslash\\
 & exp(-s(w_{12345}\leftarrow w_{123}w_{45})),exp(-s(w_{12345}\leftarrow w_{1234}w_{5}))]\end{aligned}$$

and

$$\theta[1,5,1:4]=[0,1,0,0]$$

Now, $(\theta[ij]\circ S_{ij})$ will be $$(\theta[i,j]\circ S_{ij})=[0,exp(-s(w_{12345}\leftarrow w_{12}w_{345})),0,0]$$

$\bar{I}_{l}\circ\Pi_{l}$, $\Pi[i]_{l}^{T}$ and $(\bar{I}_{l}\circ\Pi_{l}^{i})\Pi[i]_{l}^{T}$ then would be $$\bar{I}_{l}\circ\Pi_{l}=\left[\begin{array}{cccc}
\pi(w_{2345}) & 0 & 0 & 0\\
0 & \pi(w_{345}) & 0 & 0\\
0 & 0 & \pi(w_{45}) & 0\\
0 & 0 & 0 & \pi(w_{5})
\end{array}\right],$$

$$\Pi[i]_{l}^{T}=\left[\begin{array}{c}
\pi(w_{1})\\
\pi(w_{12})\\
\pi(w_{123})\\
\pi(w_{1234})
\end{array}\right],$$

and $$(\bar{I}_{l}\circ\Pi_{l}^{i})\Pi[i]_{l}^{T}=\left[\begin{array}{c}
\pi(w_{1})\pi(w_{2345})\\
\pi(w_{12})\pi(w_{345})\\
\pi(w_{123})\pi(w_{45})\\
\pi(w_{1234})\pi(w_{5})
\end{array}\right]$$

Finally, $\pi(w_{12345})$ would be $$\pi(w_{12345})=exp(-s(w_{12345}\leftarrow w_{12}w_{345}))\pi(w_{12})\pi(w_{345})$$

In the formulation above, the tree structure is codified in tensor $\theta$ and doesn’t need a dynamic network to compute $s(W,t)$, hence this formulation can be used for batch computation.

Batched Recursive Neural Network
--------------------------------

Let $B$ be the batch of size $b$, $N$ be the length of the longest sentence in batch $B$. Let $\tilde{\theta}$ be a $4$ dimensional tensor with dimensions $N\times N\times N\times b$ and $\tilde{\Pi}$ be a $N\times N\times b$ tensor.

Let $W_{m}$ be the $m$th sentence in batch $B$ and let the length of $W_{m}$ be $n$, we define $\tilde{\theta}$ as $\tilde{\theta}[1:n,1:n,:,m]=\theta(W_{m})$ and $\tilde{\Pi}$ as $\tilde{\Pi}[1:n,1:n,m]=\Pi(W_{m})$.

Let’s now compute $\tilde{\Pi}[i,j,:]$ . Let $l=j-i$ and $\tilde{I}_{l}$ is a **$l\times l\times b$** tensor with $\tilde{I}_{l}[:,:,m]=\bar{I}_{l}$, we can compute $\tilde{\Pi}[i,j,:]$ as:

$$\tilde{\Pi}[i,j,:]=(\tilde{\theta}[i,j,1:l,m]\circ\tilde{S}_{ij}[:])(\tilde{I}_{l}[:]\circ\tilde{\Pi}_{l}^{i}[:])\tilde{\Pi}[i,:]_{l}^{T}$$

Here, $\tilde{S}_{ij}[m]=S_{ij}(W_{m})$.

Implementation Details
----------------------

Assuming the code is implemented on GPU.

Let’s start by computing $(\tilde{I}_{l}[:]\circ\tilde{\Pi}_{l}^{i}[:])\tilde{\Pi}[i,:]_{l}^{T}$.

Let $\tilde{\Pi}_{l}^{i}$ be $l\times l\times b$ tensor such that $\tilde{\Pi}_{l}^{i}=\tilde{\Pi}[i+1:(i+1+l),1:l,:]$, $\tilde{\Pi}_{i}$ be $l\times b$ matrix such that $\tilde{\Pi}_{i}=\tilde{\Pi}[i,1:l,:]$ and $\Pi_{out}$ be a $b$ length vector that contains the output of $(\tilde{I}_{l}[:]\circ\tilde{\Pi}_{l}^{i}[:])\tilde{\Pi}[i,:]_{l}^{T}$ operation.

 insideKernel($\tilde{\Pi}_{l}^{i}$,$\tilde{\Pi}_{i}$,$\Pi_{out}$, l):

b = blockIdx

t= threadIdx

$\tilde{\Pi}_{l}^{i}$[\[]{}t,l-t,b[\]]{}[\*]{}$\tilde{\Pi}_{i}$[\[]{}t,b[\]]{}

 

Let’s define a compacting operation which given a vector of one-hot vectors, returns an array of indexes that were 1 in the input matrix. Let $\tilde{\theta}_{ij}$ be $l\times b$ one hot matrix such that $\tilde{\theta}_{ij}=\theta[i,j,1:l,:]$ and $\theta_{ij}^{out}$ is a length $b$ vector for output, we define an kernel compactIdxKernel as

compactIdxKernel($\tilde{\theta}_{ij}$,$\theta_{ij}^{out}$)

Let’s define a scoringKernel which takes in the length $d$ embedding vectors $X_{i..k}$ and $X_{k+1..j}$, label $T_{i..j}$ and return $w_{i..j}$’s embedding vector $X_{i..j}$and its score $s_{i..j}$.

Let $\tilde{X}$ be a $N\times N\times b\times d$ tensor such that $\tilde{X}[i,j,m]$ is the embedding vector for span $w_{i..j}^{m}$ from sentence $m$ in batch $B$, let $\tilde{T}_{ij}$ be a $N\times N\times b$ label matrix such that $T[i,j,m]=T_{i...j}^{m}$ is the label for span $w_{i..j}^{m}$ from sentence $m$ in batch $B$ and $S_{ij}$ is a length $b$ vector such that $S_{ij}[m]=s(w_{i..j}^{m})$ i.e. $m$th index in $S_{ij}$ is the score of span $w_{i..j}^{m}$ from sentence $m$ in batch $B$.

Using these defintions, we now define another kernel scoringKernelBatch which takes as input tensors $\tilde{X}$, $\tilde{T}$, $\theta_{ij}^{out}$ and returns $S_{ij}$

scoringKernelBatch($\tilde{X}$,i,j,$\tilde{T}$,$\theta_{ij}^{out}$, $S_{ij}$):

b = batchIdx

k = $\theta_{ij}^{out}$[\[]{}b[\]]{}

scoringKernel($\tilde{X}$[\[]{}i,k,b[\]]{}, $\tilde{X}$[\[]{}k+1,j,b[\]]{},$\tilde{T}$[\[]{}i,j,b[\]]{},$\tilde{X}$[\[]{}i,j,b[\]]{}, $S_{ij}$[\[]{}b[\]]{})

 

Finally, we define insideKernel that computes $\tilde{\Pi}[i,j,:]$

insideMethod(i,l,$\tilde{\Pi},$$\tilde{\theta}$,$\tilde{T}$,$\tilde{X}$ b):

$\tilde{\Pi}_{l}^{i}$=$\tilde{\Pi}$[\[]{}i+1:(i+l+1),1:l,:[\]]{}

$\tilde{\Pi}_{i}$=$\tilde{\Pi}$[\[]{}i,1:l,:[\]]{}

$\Pi_{out}$= zeros(b)

insideKernel($\tilde{\Pi}_{l}^{i}$,$\tilde{\Pi}_{i}$,$\Pi_{out}$)

$\tilde{\theta}_{ij}$= $\theta[i,j,:,:]$

$\theta_{ij}^{out}$ = zeros(b)

compactIdxKernel($\tilde{\theta}_{ij}$,$\theta_{ij}^{out}$)

$S_{ij}$ = zeros(b) 

scoringKernelBatch($\tilde{X}$,i,j,$\tilde{T}$,$\theta_{ij}^{out}$, $S_{ij}$)

$\Pi[i,j,:]=S_{ij}\circ\Pi_{out}$

[^1]: For simplicity, we index all our matrices starting at 1, so first element of matrix $\theta(i,j,k)$ would be $\theta[1,1,1]$.

[^2]: In similar vein, the first element of matrix $\Pi(W)$ would be indexed at $[1,1]$.
