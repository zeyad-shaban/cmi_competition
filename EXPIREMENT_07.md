LIGHTGMB with simple model

Base:
    agg_recipe = {
        "gesture": ["first"],
        "subject": ["first"],
        
        # ---Rotation
        "rotvec_x": ["mean", "std"],
        "rotvec_y": ["mean", "std"],
        "rotvec_z": ["mean", "std"],
        "rot_angle": ["mean", "std"],
        # angular velocity
        "rotvec_x_diff": ["mean", "std", "min", "max"],
        "rotvec_y_diff": ["mean", "std", "min", "max"],
        "rotvec_z_diff": ["mean", "std", "min", "max"],
        "angular_mag": ["mean", "std", "max"], # dont' add min here, it was always zero
        
        # ---Accelrometer
        "acc_x": ["mean", "std"],
        "acc_y": ["mean", "std"],
        "acc_z": ["mean", "std"],
        # mag
        "acc_mag": ["mean", "std"],
        # jerk
        "jerk_acc_x": ["mean", "std", "min", "max"],
        "jerk_acc_y": ["mean", "std", "min", "max"],
        "jerk_acc_z": ["mean", "std", "min", "max"],
        # fft stuff
        "fft_acc_x": ["mean", "std"],
        "fft_acc_y": ["mean", "std"],
        "fft_acc_z": ["mean", "std"],
    }
    
    macro: 0.445, binary: 0.928, competition: 0.686
    
With LightGBM macro: 0.466, binary: 0.925, competition: 0.696



