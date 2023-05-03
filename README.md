# A Deep Item Response Theory Model Based on Transformer Neural Networks
This project attempts to implement a deep learning model based on the Rasch model, where item difficulty is represented using sentence embeddings.

## Required Packages:
1. PyTorch
2. SBERT
3. mirt (for the baseline R model)
4. wandb (for logging)
5. pytorch-lightning
6. datasets

## Purpose of Each File:
1. DIRT.py: Code for the deep IRT model. It is based on another deep IRT model of the same name, but note that I do not attempt an exact replication.
2. classical_estimation.R: code for fitting a classical Rasch model to the data.

## The Data

The model was tested using item responses to a fraction test administered to Taiwanese students. Citation information is below. Note that you will need to contact the authors to get access to the data, I do not have permission to share.

## Citations:

``
@article{chen2015assessment,
  title={Assessment of Taiwanese Students’ Conceptual Development of Fractions},
  author={Chen, Yi-Hsin and Leu, Yuh-Chyn and Pride, Bryce L and Chavez, Teresa},
  journal={Journal of Education and Human Development},
  volume={4},
  number={2},
  pages={10--21},
  year={2015}
}
``
