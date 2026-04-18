import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset with memory optimization.
    """
    dtype_dict = {f"V{i}": "float32" for i in range(1, 29)}
    dtype_dict.update({
        "Time": "float32",
        "Amount": "float32",
        "Class": "int8"
    })

    df = pd.read_csv(filepath, dtype=dtype_dict)
    
    return df