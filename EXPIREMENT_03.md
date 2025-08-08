# Combined Accelrometer and Rotation Readings
Base:
    gesture: first
    subject: first
    # ---Rotation
    rotvec_x: mean std
    rotvec_y: mean std
    rotvec_z: mean std
    rot_angle: mean std
    # angular velocity
    rotvec_x_diff: mean std min max
    rotvec_y_diff: mean std min max
    rotvec_z_diff: mean std min max
    angular_mag: mean std max
    
    # ---Accelrometer
    acc_x: mean std
    acc_y: mean std
    acc_z: mean std
    # mag
    acc_mag: mean std
    # jerk
    jerk_acc_x: mean std min max
    jerk_acc_y: mean std min max
    jerk_acc_z: mean std min max
    # fft stuff
    fft_acc_x: mean std
    fft_acc_y: mean std
    fft_acc_z: mean std

Summary Of Result:
0.68

## Hyphothesis 1: Adding Demographics will improve model
adult_child                                                 0.68
adult_child + handeness + shoulder_to_wrist_cm              0.67
shoulder_to_wrist_cm                                        0.67
age                                                         0.68

!Huh, they.. don't?

## Hypothesis 2: training only on adults (adult_child=1) will improve the results
get those fucken kids out of here                           0.67

Huh???
## Hypothesis 2.1: trianing only on kids = better (addult_child=0)
got those fucken adults out of heree                        0.66