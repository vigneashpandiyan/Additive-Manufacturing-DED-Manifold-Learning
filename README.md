# Additive-Manufacturing-DED-Manifold-Learning
Monitoring of direct energy deposition process using deep-net based manifold learning and co-axial melt pool imaging
# Journal link
https://www.sciencedirect.com/science/article/pii/S2213846322001274?via%3Dihub
# Overview
![Experimental](https://github.com/vigneashpandiyan/Additive-Manufacturing-Self-Supervised-Bayesian-Representation-Learning-Acoustic-Emission/assets/39007209/ae39aefb-d74b-4507-87e5-991a0c0cb819)

Real-time Artificial Intelligence (AI) based monitoring systems are on the rise as an alternative to post-processing inspection techniques with advances in sensing techniques and Deep Learning (DL). The article focuses on how the manifolds learnt by the embedded space of the two convolutional generative models, such as autoencoder, and Generative Adversarial Networks (GAN), could be exploited to differentiate built conditions. The co-axially mounted CCD camera acquired the melt pool morphology corresponding to six build parameters covering the process map from Lack of Fusion (LoF) to conduction regime. The images acquired from the CCD camera constituted the dataset to train and test the model performance. After training, the latent space of both the networks would have captured the commonality and differences, i.e. unique manifolds of the melt pool morphologies corresponding to the six build conditions. The learned manifolds from the two trained Convolutional Neural Networks (CNN) models were exploited by combining with One-Class SVM to classify the ideal build quality from the other conditions supervisedly. The prediction of the trained One-class SVM on the two latent spaces of the CNN models had an overall classification accuracy of ≈97%. The results on the proposed methodology demonstrate the potential and robustness of the developed vision-based methodology using manifold learning for DED process monitoring.


![Experimental](https://github.com/vigneashpandiyan/Additive-Manufacturing-Self-Supervised-Learning-Coaxial-DED_Process-Zone-Imaging/assets/39007209/92c60c07-c014-4a47-9644-930fac822a55)


# Manifold Learning

![Abstract](https://github.com/vigneashpandiyan/Additive-Manufacturing-Self-Supervised-Learning-Coaxial-DED_Process-Zone-Imaging/assets/39007209/fb656011-2804-4525-8fae-d6c1a206e61d)
 ![Result](https://github.com/vigneashpandiyan/Additive-Manufacturing-DED-Manifold-Learning/assets/39007209/4b7d15be-b799-4798-bd27-4e7949ba9771)

# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-DED-Manifold-Learning
cd Additive-Manufacturing-DED-Manifold-Learning
python Main_AutoEncoder.py
python Main_GANomaly.py
```

# Citation
```
@article{pandiyan2022monitoring,
  title={Monitoring of Direct Energy Deposition Process Using Manifold Learning and Co-Axial Melt Pool Imaging},
  author={Pandiyan, Vigneashwara and Cui, Di and Parrilli, Annapaola and Deshpande, Pushkar and Masinelli, Giulio and Shevchik, Sergey and Wasmer, Kilian},
  journal={Manufacturing Letters},
  volume={33},
  pages={776--785},
  year={2022},
  publisher={Elsevier}
}
```
