---
layout: page
title: Selected Projects
---

#### *Compositional Language Modeling* ####
Implemented the idea proposed in the paper "*Compositional Approach to Language Modeling*". The code was written in Java. We used Nd4j as our main matrix library and UJMP to store Inside-Outside scores due to its support for sparse matrices. For Inside-Outside score calculation we used the grammar from the Stanford parser. The code is written in a modular way to allow plugging in other grammars without much effort.

[[pdf](https://arxiv.org/abs/1604.00100v1.pdf), [code](http://github.com/kushalarora/CompositionalLM.git)]

#### *Sentence Level Recurrent Neural Network* ####
Implementation of Sentence Level RNN described in "*Contrastive Entropy: A new evaluation metric for unnormalized models*". The implementation was done using Theano and Numpy.

[[pdf](http://arxiv.org/pdf/1601.00248v2.pdf), [code](http://github.com/kushalarora/sentenceRNN.git)]

#### *Comparative Evaluation of Manifold Learning Algorithms* ####
Implemented the state of the art dimensionality reduction algorithms in python using Scipy and compared them on
four data sets, namely RaceSpace, Digits, Faces and Swiss Roll. The project was an individual effort and done as a course project for Advanced Machine Learning class.

<em><strong>Algorithms implemented</strong>: Local Linear Embedding, ISOMap, Laplacian Eigenmaps, Hessian LLE, Local Tanget Space Analysis, Stochastic Neighborhood Embedding</em>


[[pdf](/assets/AMLProjectReport.pdf), [code](https://github.com/kushalarora/ManifoldAlgorithms.git)]

#### Comparative Evaluation of Supervised Learning Algorithms ####
Built a generic framework to run a list of Supervised Learning Algorithms in Python using scikit­-learn and Theano. The framework was used to do a comparative study on following data sets: Wisconsin Breast Cancer, Iris, Higgs, OCR and Hand Writing Recognition across a range of supervised learning algorithms. This project was done in a team of three for Machine Learning class.

<em><strong>Algorithms evaluated</strong>: Multi Layer Perceptron, Stacked Auto Encoders, Deep Belief Network, Support Vector Machine, Random Forest, Decision Tree, AdaBoost Decision Tree. </em>

[[code](https://github.com/kushalarora/SupervisedMLAlgorithms.git)]

#### *Ontology Alignment for Knowledge Bases* ####
Implemented and evaluated PARIS, an ontology alignment technique that uses web text based interlingua for aligning relations and entities. Ontologies for Freebase, NELL and Yago were mapped to each other using label propagation algorithm. This project was done as a independent study in Data Science Lab under Dr. Daisy Wang and was a part of larger objective to build a master KB for the lab.


[[pdf](/assets/pidgin.pdf), [code](https://github.com/kushalarora/pidgin.git)]
