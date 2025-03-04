# network_train
A simple neural network written in c++ and was trained by MNIST,with an accuracy of around 88%.But I currently can't find any further ways to improve the accuracy.  

## Learning rules

<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;\large&space;\omega\rightarrow&space;\omega&space;-&space;\eta\frac{\partial&space;C}{\partial&space;\omega}" title="\large \omega\rightarrow \omega - \eta\frac{\partial C}{\partial \omega}" />

<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;\large&space;b\rightarrow&space;b-&space;\eta\frac{\partial&space;C}{\partial&space;b}" title="\large b\rightarrow b- \eta\frac{\partial C}{\partial b}" />

## Input layer cost vector

<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\LARGE&space;\delta^L&space;=\triangledown_a&space;C&space;\odot&space;{\sigma}'(z^L)" title="\LARGE \delta^L =\triangledown_a C \odot {\sigma}'(z^L)" />

## Intermediate layer cost vector

<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\LARGE&space;\delta^l&space;=((\omega&space;^{l&plus;1})^T)\delta&space;^{l&plus;1})\odot&space;\sigma'(z^l)" title="\LARGE \delta^l =((\omega ^{l+1})^T)\delta ^{l+1})\odot \sigma'(z^l)" />

## Biases value modification amount

<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\LARGE&space;\frac{\partial&space;C}{\partial&space;b^l_j}=\delta&space;^l_j" title="\LARGE \frac{\partial C}{\partial b^l_j}=\delta ^l_j" />

## Weights value modification amount

<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\LARGE&space;\frac{\partial&space;C}{\partial&space;w&space;^l_{jk}}=a^{l-1}_k&space;\delta&space;^l_j" title="\LARGE \frac{\partial C}{\partial \omega ^l_{jk}}=a^{l-1}_k \delta ^l_j" />

