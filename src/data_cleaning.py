import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# Clean DataFrame
def get_counts(df) -> pd.DataFrame:
    return df.groupby("sequence_id")["sequence_counter"].max() - df.groupby("sequence_id")["sequence_counter"].min() + 1


def log_dropped(len_before, len_after, description):
    dropped = len_before - len_after
    perc_dropped = (dropped / len_before) * 100
    print(f"[{description}] Remaining: {len_after}/{len_before} " f"(-{dropped}, {perc_dropped:.1f}% dropped)")

def remove_gravity(df: pd.DataFrame, quat_cols=("rot_x", "rot_y", "rot_z", "rot_w"), acc_cols=("acc_x", "acc_y", "acc_z"), gravity_mag=9.81):
    """Return a copy with added linear_acc_x/y/z and linear_acc_mag.
    Writes only to rows with valid quaternion; invalid rows keep original accel.
    Expects quat order (x,y,z,w).
    """
    df = df.copy()

    quat_df = df[list(quat_cols)]
    # valid if no NaNs and not all zeros (tolerance for floating)
    valid_mask = (~quat_df.isnull().any(axis=1)) & (~np.isclose(quat_df.values, 0).all(axis=1))
    valid_idx = df.index[valid_mask]

    # By default keep original accel values for invalid rows
    df["linear_acc_x"] = df[acc_cols[0]].astype(float)
    df["linear_acc_y"] = df[acc_cols[1]].astype(float)
    df["linear_acc_z"] = df[acc_cols[2]].astype(float)

    if len(valid_idx) > 0:
        quat_vals = quat_df.loc[valid_idx].to_numpy(dtype=float)  # shape M x 4 (x,y,z,w)
        accel_vals = df.loc[valid_idx, list(acc_cols)].to_numpy(dtype=float)  # shape M x 3

        rotations = R.from_quat(quat_vals)  # expects [x,y,z,w]
        accel_world = rotations.apply(accel_vals, inverse=True)  # sensor -> world

        gravity = np.array([0.0, 0.0, gravity_mag], dtype=float)
        linear_world = accel_world - gravity  # linear accel in world frame

        # write results back only to valid rows
        df.loc[valid_idx, "linear_acc_x"] = linear_world[:, 0]
        df.loc[valid_idx, "linear_acc_y"] = linear_world[:, 1]
        df.loc[valid_idx, "linear_acc_z"] = linear_world[:, 2]

    df["linear_acc_mag"] = np.linalg.norm(df[["linear_acc_x", "linear_acc_y", "linear_acc_z"]].to_numpy(), axis=1)
    return df




def normalize_sequence_count(df: pd.DataFrame, group_col: str = "sequence_id", seq_counter_col: str = "sequence_counter") -> pd.DataFrame:
    """
    Adds gesture_counter_length with a value of the length of the elmeents of teh array
    it gets maximum + 1, so it is the execlusive end
    """
    out_groups = []
    max_count = get_counts(df).max()

    for _, group in df.groupby(group_col, sort=False):
        g = group.copy()
        # compute length for this group
        group_count = g[seq_counter_col].max() - g[seq_counter_col].min() + 1
        needed = max_count - group_count

        if needed > 0:
            last = g.iloc[-1].copy()
            repeats = pd.DataFrame([last] * needed)

            # continue the sequence_counter for appended rows
            start = int(last[seq_counter_col])
            repeats[seq_counter_col] = range(start + 1, start + 1 + needed)

            g = pd.concat([g, repeats], ignore_index=True)

        out_groups.append(g)
    return pd.concat(out_groups, ignore_index=True)


def clean_df(df: pd.DataFrame, drop_rot_na=False, drop_thm_na=False, min_gesture_count=26, max_gesture_count=38):
    """
    min_gesture_count = 28, max_gesture_count = 35 was the current best performing
    put as -1 to not do it
    """
    df = df.copy()
    target_gestures = df[df["sequence_type"] == "Target"]["gesture"].unique()

    filtered_df = df[df["phase"] == "Gesture"]

    curr_len = len(filtered_df)
    if drop_rot_na:
        # drop na rotation
        bad_seq_id = filtered_df[filtered_df["rot_w"].isnull()]["sequence_id"].unique()
        bad_seq_mask = filtered_df["sequence_id"].isin(bad_seq_id)
        filtered_df = filtered_df[~bad_seq_mask]
        
        log_dropped(curr_len, len(filtered_df), "rot_na")
        curr_len = len(filtered_df)

    if drop_thm_na:
        for i in range(1, 6):
            bad_seq_id = filtered_df[filtered_df[f"thm_{i}"].isnull()]["sequence_id"].unique()
            bad_seq_mask = filtered_df["sequence_id"].isin(bad_seq_id)
            filtered_df = filtered_df[~bad_seq_mask]


        log_dropped(curr_len, len(filtered_df), "thm_na")
        curr_len = len(filtered_df)

    # drop outliers in terms of count
    if min_gesture_count != -1 and max_gesture_count != -1:
        gesture_counts = get_counts(filtered_df)
        valid_mask = (gesture_counts >= min_gesture_count) & (gesture_counts <= max_gesture_count)
        valid_idx = gesture_counts[valid_mask].index
        filtered_df = filtered_df[filtered_df["sequence_id"].isin(valid_idx)]


        log_dropped(curr_len, len(filtered_df), "gesture_len outliers")
        curr_len = len(filtered_df)
    
    filtered_df = filtered_df.ffill().bfill().fillna(0)
    filtered_df = remove_gravity(filtered_df)
    filtered_df = normalize_sequence_count(df)
    
    return filtered_df, target_gestures


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "sequence_id": [1, 1, 1, 2, 2],
            "sequence_counter": [0, 1, 2, 0, 1],
            "phase": ["Gesture"] * 5,
            "sequence_type": ["Target"] * 3 + ["Non-Target"] * 2,
            "gesture": ["A"] * 3 + ["B"] * 2,
            "rot_w": [1] * 5,
            "thm_1": [1] * 5,
            "thm_2": [1] * 5,
            "thm_3": [1] * 5,
            "thm_4": [1] * 5,
            "thm_5": [1] * 5,
        }
    )

    cleaned = clean_df(df)
