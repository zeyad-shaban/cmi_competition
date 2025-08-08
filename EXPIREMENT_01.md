# Using acc_x, acc_y, acc_z

## Hypothesis 1: More features = better
N: Base is acc_x, acc_y, acc_z                  After Normalization
- mean:                                0.45     0.46
- mean + std                           0.61
* mean + std + min                     0.62
- mean + std + min + max               0.62
- mean + std + min + max + p25 + p75   0.62     0.61

## Hyptothesis 2: if i added more neurons the model will pick more complex pattern
- mean + std + min + max + p25 + p75   0.61
-> Hypothesis DEBUNKED


## Hypothesis 3: Using FFT would be better
- fft_mean:                              0.19     0.35
- fft_mean, fft_std                               0.47
- fft_mean, fft_std, fft_max                      0.44
* fft_mean, fft_std, fft_min                      0.47
- fft_mean, fft_std, fft_max, fft_min:   0.03     0.48

fft_max introduces confusion

## Hypothesis 3.1 Addding Accelrometer magnitude is amazing
mean std, mag_mean, mag_std                       0.63
mean std, mag_mean, mag_std, mag_min, mag_max     0.62

## Hypothesis 3.2: Adding Jerk = Good
B Base: mean, std, fft_mean, fft_std, mag_mean, mag_std:    0.63
jerk_mean, jerk_std:                                        0.64
jerk_mean, jerk_std, jerk_min, jerk_max                     

! JERK_MIN, JERK_MAX HELPS ALOT TO FIND CLASS NON-TARGET:

## Hypothesis 3.5: combining fft and raw accelorometer = ultimate good
mean, std, min, max                               0.62
mean, std, min                                    0.62
mean, std                                         0.62

mean, std, min, max, fft_mean, fft_std            0.63
mean, std, fft_mean, fft_std                      0.62
mean, std, mag_mean, mag_std, fft_mean, fft_std   0.63

## Hypothesis 3: Using an LSTM Layer and feeding all data is better

## Hypothesis 3.5: Using combination of LSTM and other is better

## Hypothesis 5: Transition phase is unimportant

## Hypothesis 6: Adding all Non-Target elements for the model to train on is better than a generic Non-Target
Hypothesis delayed as binary F1 = 90%

## Hypothesis 7: feed transition phase





Hyptohesis Later:
discritize R^3 -> R^1
Does sex matter?
Does handiness matter?