import pandas as pd
import re

def extract_frame_index(frame_key: str) -> int:
    """
    Extract numeric x from 'frame_x_face00' or similar 'frame_<number>_*' patterns.
    Falls back to the first number found if the preferred pattern isn't present.
    Returns -1 if no number is found.
    """
    if not isinstance(frame_key, str):
        return -1
    m = re.search(r'frame[_-]?(\d+)', frame_key)
    if m:
        return int(m.group(1))
    m2 = re.search(r'(\d+)', frame_key)
    return int(m2.group(1)) if m2 else -1

def sort_inference_csv(csv_path: str, output_csv_path: str = None) -> pd.DataFrame:
    """
    Read the CSV and return a DataFrame sorted by:
      - video_id
      - frame_index (extracted from frame_key)
      - eye_side (left first, then right)
      - frame_key (as tie-breaker)
    If output_csv_path is provided, writes the sorted CSV to that path.
    """
    df = pd.read_csv(csv_path)

    # Extract numeric frame index from frame_key
    df['frame_index'] = df['frame_key'].apply(extract_frame_index)

    # Normalize eye_side and set ordering: left before right
    df['eye_side_norm'] = df['eye_side'].astype(str).str.strip().str.lower()
    df['eye_order'] = df['eye_side_norm'].map({'left': 0, 'right': 1}).fillna(2)

    # Sort
    df_sorted = df.sort_values(by=['video_id', 'frame_index', 'eye_order', 'frame_key']).reset_index(drop=True)

    # Optionally write out
    if output_csv_path:
        df_sorted.to_csv(output_csv_path, index=False)

    # Drop helper columns before returning (optional)
    df_sorted = df_sorted.drop(columns=['frame_index', 'eye_side_norm', 'eye_order'])

    return df_sorted

# Example usage:
sorted_df = sort_inference_csv(
    csv_path="/inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/inference_pairs.csv",
    output_csv_path="/inwdata2a/sudhanshu/landmarks_only_training/inference_pipeline/inference_pairs-sorted.csv"
)
print(sorted_df.head())