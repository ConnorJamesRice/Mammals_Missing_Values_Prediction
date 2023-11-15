# Mammals Missing Values Prediction / PanTHERIA_MSW05_Mammals ALS Matrix Completion

This repository contains Python code for implementing Alternating Least Squares (ALS) Matrix Completion for the PanTHERIA dataset. The code performs data extraction, matrix factorization, cross-validation, hyperparameter tuning, and error analysis.

## Code Structure

- **`main.py`**: Entry point of the code. It orchestrates the data extraction, ALS matrix factorization, hyperparameter tuning, and error analysis.
- **`data_extract.py`**: Defines a class `data_extract` responsible for reading and processing the PanTHERIA dataset.
- **`ALS_Matrix_Completion.py`**: Implements the ALS Matrix Completion algorithm and cross-validation methods.

## Implementation Overview

### Data Extraction

- `data_extract.py`:
    - Utilizes Pandas to read the PanTHERIA dataset (`PanTHERIA_WR05_mammals.txt`).
    - Implements methods to extract and process specific data based on orders within the dataset.
    - Splits data into order subsets where each column with unknown information is removed
    - Splits the data into training and testing sets.

### ALS Matrix Completion

- `ALS_Matrix_Completion.py`:
    - Implements the ALS algorithm for matrix factorization.
    - Defines methods to perform cross-validation, calculate mean squared error (MSE), and perform hyperparameter tuning using a 'drunken sailor' (random walk) search approach.

### Hyperparameter Tuning

- Employs a 'drunken sailor' search strategy to tune hyperparameters (`lam_U` and `lam_V`) by evaluating performance through cross-validation.

## How to Use

### Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

1. Open a terminal and navigate to the project directory.
2. Run the `main.py` file:

    ```bash
    python main.py
    ```

## Explanation and Thought Process

- **Data Processing**: The `data_extract` class handles dataset processing, focusing on specific orders within the PanTHERIA dataset.
- **Matrix Completion**: Utilizes ALS algorithm for matrix factorization, aiming to predict missing values in the dataset.
- **Cross-Validation**: Implements k-fold cross-validation to assess model performance and tune hyperparameters effectively.
- **Hyperparameter Tuning**: Employs a 'drunken sailor' search strategy to explore hyperparameter space for optimal model performance.

## Contributing and License

Feel free to contribute by opening issues or submitting pull requests. This project is licensed under the [MIT License]([LICENSE](https://esapubs.org/archive/ecol/E090/184/metadata.htm)).

## Authors

- [Your Name](https://github.com/ConnorJamesRice)
