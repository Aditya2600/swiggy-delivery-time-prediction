import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

# Target column
TARGET = "time_taken"

# ---------------- Logger Setup ----------------
logger = logging.getLogger("data_preparation")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

# ---------------- Utility Functions ----------------
def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        logger.error(f"The file {data_path} does not exist")
        raise


def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    """Split the dataframe into train and test sets."""
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    return train_data, test_data


def read_params(file_path: Path):
    """Read parameters from params.yaml."""
    with open(file_path, "r") as f:
        params_file = yaml.safe_load(f)
    return params_file


def save_data(data: pd.DataFrame, save_path: Path) -> None:
    """Save dataframe to CSV."""
    data.to_csv(save_path, index=False)


# ---------------- Main Script ----------------
if __name__ == "__main__":
    # Root path
    root_path = Path(__file__).parent.parent.parent

    # Data paths
    data_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv"
    save_data_dir = root_path / "data" / "interim"
    save_data_dir.mkdir(exist_ok=True, parents=True)

    # Train/test filenames
    train_filename = "train.csv"
    test_filename = "test.csv"

    save_train_path = save_data_dir / train_filename
    save_test_path = save_data_dir / test_filename

    # Params file
    params_file_path = root_path / "params.yaml"

    # Load cleaned data
    df = load_data(data_path)
    logger.info("Data loaded successfully")

    # Read parameters
    parameters = read_params(params_file_path)["Data_Preparation"]
    test_size = parameters["test_size"]
    random_state = parameters["random_state"]
    logger.info("Parameters read successfully")

    # Split into train and test
    train_data, test_data = split_data(df, test_size=test_size, random_state=random_state)
    logger.info("Dataset split into train and test data")

    # Save datasets
    data_subsets = [train_data, test_data]
    data_paths = [save_train_path, save_test_path]
    filename_list = [train_filename, test_filename]

    for filename, path, data in zip(filename_list, data_paths, data_subsets):
        save_data(data=data, save_path=path)
        logger.info(f"{filename.replace('.csv', '')} data saved to {path}")