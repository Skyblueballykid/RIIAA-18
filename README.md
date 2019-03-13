# RIIAA Escuela18

My edits and code related to the Higgs Boson dataset Deep Learning competition of the 2018 International Artificial Intelligence meeting at UNAM

My final submission consisted of:
- 7 hidden layers with 80 units each
- A truncated Normal distribution kernel
- ELU activation
- Batch size of 28
- Adam optimizer (outperformed Powersign for this configuration)

The model accuracy consistently topped out at around 170 epochs of training time. 

The IAmigos_RIIAA_Reto_Machine_Learning.ipynb file was the winning submission from another team. This model used Sklearn's GBClassifier. 
