import pandas as pd


# Clean DataFrame
def get_counts(df) -> pd.DataFrame:
    return df.groupby("sequence_id")["sequence_counter"].max() - df.groupby("sequence_id")["sequence_counter"].min() + 1


def log_dropped(len_before, len_after, description):
    dropped = len_before - len_after
    perc_dropped = (dropped / len_before) * 100
    print(f"[{description}] Remaining: {len_after}/{len_before} " f"(-{dropped}, {perc_dropped:.1f}% dropped)")


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

        # Those had no impact on the results, will look at them later
        # broadcast the group length to existing rows
        # g["gesture_counter_length"] = group_count
        # g["is_original"] = 1

        if needed > 0:
            last = g.iloc[-1].copy()
            repeats = pd.DataFrame([last] * needed)

            # continue the sequence_counter for appended rows
            start = int(last[seq_counter_col])
            repeats[seq_counter_col] = range(start + 1, start + 1 + needed)

            # ensure the appended rows also carry the same group length
            # repeats["gesture_counter_length"] = group_count
            # repeats["is_original"] = 0

            g = pd.concat([g, repeats], ignore_index=True)

        out_groups.append(g)

    return pd.concat(out_groups, ignore_index=True)


def clean_df(df: pd.DataFrame, drop_rot_na=True, drop_thm_na=True, min_gesture_count=28, max_gesture_count=35):
    """
    min_gesture_count = 28, max_gesture_count = 35 was the current best performing
    put as -1 to not do it
    """
    non_target_gestures = df[df["sequence_type"] == "Non-Target"]["gesture"].unique()
    target_gestures = df[df["sequence_type"] == "Target"]["gesture"].unique()

    filtered_df = df[df["phase"] == "Gesture"]
    filtered_df.loc[filtered_df["sequence_type"] == "Non-Target", "gesture"] = non_target_gestures[0]

    curr_len = len(df)
    if drop_rot_na:
        # drop na rotation
        bad_seq_id = df[df["rot_w"].isnull()]["sequence_id"].unique()
        bad_seq_mask = filtered_df["sequence_id"].isin(bad_seq_id)
        filtered_df = filtered_df[~bad_seq_mask]
        
        log_dropped(curr_len, len(filtered_df), "rot_na")
        curr_len = len(filtered_df)

    if drop_thm_na:
        for i in range(1, 6):
            bad_seq_id = df[df[f"thm_{i}"].isnull()]["sequence_id"].unique()
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
