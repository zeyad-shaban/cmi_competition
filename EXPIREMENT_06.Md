How to deal with the >30 outliers?

## Method 1:
Remove <= 20 (33 sequence)
remove > 50 (32 sequence)
For rest expand to 50 by adding zeros and adding feature for the acutal end time

Base Score: 
features_of_interest = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 'is_original']
macro: 0.31, binary: 0.89, competition: 0.60


## Using the rotation eigenvectors base 
features_of_interest = ['rotvec_x', 'rotvec_y', 'rotvec_z', 'rot_angle', 'is_original']
macro: 0.30, binary: 0.88, competition: 0.59

## Adding Accelrometer base features (and fft)
features_of_interest = ['rotvec_x', 'rotvec_y', 'rotvec_z', 'rot_angle', 'fft_acc_x', 'fft_acc_y', 'fft_acc_z', 'acc_mag', 'is_original']
macro: 0.32, binary: 0.90, competition: 0.61

## 3 Adding all handcrafted features
"is_original", "rotvec_x", "rotvec_y", "rotvec_z", "rot_angle", "rotvec_y_diff", "rotvec_x_diff", "rotvec_z_diff", "angular_mag", "fft_acc_x", "fft_acc_y", "fft_acc_z", "acc_mag", "jerk_acc_x", "jerk_acc_y", "jerk_acc_z",
macro: 0.33, binary: 0.88, competition: 0.61

## instead of "is_original" have the sequence length directly
macro: 0.33, binary: 0.89, competition: 0.61

## the accelromemter fourier transfomr feature seems to be literal noise
"gesture_counter_length","acc_x","acc_y","acc_z","rotvec_x","rotvec_y","rotvec_z","rot_angle","rotvec_x_diff","rotvec_y_diff","rotvec_z_diff","angular_mag",
macro: 0.33, binary: 0.91, competition: 0.62

## the rotvec_diff is noise (i will keep angular magnitude)
"gesture_counter_length","acc_x","acc_y","acc_z","rotvec_x","rotvec_y","rotvec_z","rot_angle","angular_mag"
macro: 0.34, binary: 0.90, competition: 0.62

## removing the anguular magnitude
"gesture_counter_length","acc_x","acc_y","acc_z","rotvec_x","rotvec_y","rotvec_z","rot_angle"
macro: 0.31, binary: 0.89, competition: 0.60
!  angular magntidue is extremly important

##  having acc_magnitude as well
"gesture_counter_length","acc_x","acc_y","acc_z","rotvec_x","rotvec_y","rotvec_z","rot_angle","angular_mag", "acc_magnitude"
macro: 0.33, binary: 0.90, competition: 0.62

## adding  "shoulder_to_wrist_cm", "elbow_to_wrist_cm",
macro: 0.33, binary: 0.89, competition: 0.61

"gesture_counter_length","acc_x","acc_y","acc_z","rotvec_x","rotvec_y","rotvec_z","rot_angle","angular_mag"
## Adding thm_1, .. 5
macro: 0.38, binary: 0.91, competition: 0.64

## adding fft_acc_x, y, z
"gesture_counter_length", "acc_x", "acc_y", "acc_z", "fft_acc_x", "fft_acc_y", "fft_acc_z", # "rotvec_x", "rotvec_y", "rotvec_z", "rot_angle", "thm_1", "thm_2", "thm_3", "thm_4", "thm_5", "angular_mag",
macro: 0.42, binary: 0.91, competition: 0.67
FUCKEN YES!

## Addding acceleration magnitude
macro: 0.41, binary: 0.92, competition: 0.67
hmm odd

## Adding jerk guy
macro: 0.38, binary: 0.92, competition: 0.65
jerk was bad feature

## jerk removed, adding those and adding the rotvec_diff
macro: 0.40, binary: 0.92, competition: 0.66

## adding dropout (with rotvec_diff)
macro: 0.40, binary: 0.91, competition: 0.65
## droppout without rotvec_diff
macro: 0.41, binary: 0.91, competition: 0.66


removing dropout
## does adding orientation help?
macro: 0.39, binary: 0.92, competition: 0.66
nope

## TOF and THERMAL
score with thm_1 .. thm_5
macro: 0.37, binary: 0.91, competition: 0.64

just thm_1: macro: 0.33, binary: 0.89, competition: 0.61

no thermal readings:  macro: 0.32, binary: 0.89, competition: 0.61

now no thermal readings but having tof_1_v20
macro: 0.35, binary: 0.88, competition: 0.61

no thermal but having [tof_1_v20, tof_1_v8, tof_1_v23] trying to proof if having mroe of the save veersion of the tof is helpful or not
macro: 0.32, binary: 0.87, competition: 0.59

yes it's proofed it's just noise to include multiple of the same version

testing with those ["tof_1_v20", "tof_2_v26", "tof_3_v17", "tof_4_v5", "tof_5_v20"]
macro: 0.36, binary: 0.89, competition: 0.62

ya they barelly contain any data, or maybe i chose wrong ones?

finally enablinb only tof_1_v20, with all thermal data
macro: 0.41, binary: 0.92, competition: 0.66

without the tof_1_v20 macro
0.39, binary: 0.92, competition: 0.65

without angular_mag macro
0.41, binary: 0.92, competition: 0.66

with thm_2, thm_4, thm_5, tof_1_v20
macro: 0.43, binary: 0.90, competition: 0.66

with thm_2, thm_4, thm_5 (no tof)
macro: 0.36, binary: 0.92, competition: 0.64