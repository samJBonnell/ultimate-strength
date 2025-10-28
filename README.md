# Ultimate strength prediction of ship structures using varied machine learning techniques.

<img width="1800" height="648" alt="pod_mlp_000025" src="https://github.com/user-attachments/assets/2dcf67f0-4408-442b-b5cb-9a9a4619532e" />

## Current progress:
- Data generation defined by complete parametric ABAQUS models w/ data handling on both input and output
- POD-MLP:
    Proper Orthogonal Decomposition combined with multilayer perceptron to predict POD weights to construct a reduced stress field for simply supported variably-patch loaded, variable thickness panels
- CNN:
    Convolutional Neural Network to predict the full-order stress field of a simply supported, variably patch-loaded, variable thickness panel

## Future work:
- GNNs
- Prediction of stress with plastic materials
- Buckling prediction
