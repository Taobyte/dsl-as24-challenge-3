dsl-as24-challenge-3
==============================

AI-powered Earthquake Data Denoising

# Dataset
The dataset consists of 20'000 noise-free earthquake events and 20'000 pure noise events. The data was collected by the swiss seismological service (SED) http://seismo.ethz.ch/en/home/. 



# Butterworth

We use the bandpass butterworth filter as a baseline. We use `scipy.optimize.minimize` function to find the optimal values for the order and the lower and upper bound of the band. 

![alt text](Butterworth_orders.svg.png)

For running the training use

```
python main.py --butterworth --training True --signal_path 'path_to_signals' --noise_path 'path_to_noise'
```
. This function returns the optimized parameters `lowcut, highcut, order`. 


# Deep Denoiser

We implement the architecture from the DeepDenoiser paper https://github.com/AI4EPS/DeepDenoiser in keras and train it with the Swiss SED dataset. 

### Data Normalization
As a first step, the authors propose to normalize the earthquake event and noise timeseries by dividing by the standard deviation. 
```
normalized_eq = eq / eq.std()
normalized_noise = noise / noise.std()
```
Then the combine the noise and earthquake. 

```
noisy_eq = normalized_eq + ratio * normalized_noise
noisy_eq = noisy_eq / noisy_eq.std()
```
Then they create the ground truth masks. 

```
tmp_mask = np.abs(noisy_eq) /  
```



# Cold Diffusion

We use the model from the Cold Diffusion Model for Seismic Signal Denoising paper https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/tree/main/CDiffSD/utils. The proposed architecture uses the cold diffusion method to iteratively denoise a noisy earthquake signal. 

### Data Normalization
