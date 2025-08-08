# Using rot_x, rot_y, rot_z, rot_w
some rotation values are missing, but the competition mentions IMU should be working fine for validation, so i will drop those rows

## Hypothesis 0: figuring out the base
mean, std:                                                           0.61
mean, std, min, max                                                  0.62
## Hypothesis 1: converting system to Euler is better
just ypr mean, std                                                   0.61
just ypr mean, std, min, max                                         0.63
(ypr and rot_xyzw) mean, std, min, max                               0.62

! having both euler and quaterion is useless, and euler proved to be better
## Hyptohesis 2: converting system to eigen decomposition is better
eigenvectors + rot_angle mean, std                                   0.63
eigenvectors + rot_angle mean, std, min, max                         0.63
eigenvectors mean, std, min, max                                     0.62

!yes rot_angle is powerful

## Hypothesis 3: Euler + Eigen = something better?
eigen + euler mean, std, min, max                                    0.63
eigen mean, std + euler mean, std, min, max                          0.63

! conclusion: euler adds no further info to eigen, and eigenvector proved to be easier and far more powerful, cuz it's eigen duh fuck off

## Hypothesis 3.5: Adding Angular Velocity and its magnitude is good
### Euler Based
angular_speed mean, std                                              0.19
angular_speed, xyz mean, std                                         0.48
angular_speed, xyz mean, std, min, max                               0.49 * there is some info
eigen mean, std + angular mean, std, min, max                        0.64

## Eigen Based
angular mean, std, min, max                                          0.54
eigen mean, std + angular mean, std, min, max                        0.66

! conclusion: EIGEN IS FUCKEN GOATED BRO

## Hypothesis 3.6 Adding Angular jerk is good
angular or angular_speed means the rotvec_xyz along with its magnitude component

eigen mean, std + angular mean, std, min, max + jerk mean, std                          0.65
eigen mean, std + angular mean, std, min, max + jerk mean, std, min, max                0.65

!conclusion: jerk added nothing to the system
! conclusion: min angular_magnitude is always 0, it was useless