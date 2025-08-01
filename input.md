# Message Parsing Interface using mpi4py

This project implements a parallel, distributed ML prediction service using the Message Passing Interface (MPI). The service is designed to:

1.  Load a pre-trained scikit-learn model (`fraud_rf_model.pkl`).
2.  Prepares a scikit-learn `LabelEncoder` by fitting it on the 'status' column of a synthetic dataset (`synthetic_test_data.csv`) and saves it (to `status_label_encoder.pkl`). This encoder is used for transforming categorical data into numerical for the model. Both the model and the fitted encoder are broadcasted from the root process (rank 0) to all worker processes.
3.  Mocked Queue Interaction. The service simulates interaction with a transaction queue and a results queue.
    *   **Transaction "Queue" (Mocked Pull):** The root process reads transaction data from a CSV file (`synthetic_test_data.csv`). This file acts as the source for the mock queue. The root "pulls" batches of transactions.
    *   **Results "Queue" (Mocked Push):** After predictions are made, the root process "pushes" the results to a list, simulating sending them to a results queue.
4.  Distribute these transactions across multiple workers for parallel fraud prediction.
5.  Perform fraud prediction on each transaction using the loaded copy of the model.
6.  Gather the prediction results from all workers.
7.  Output the collected predictions and summary of operations in the root process

## Overview of Approach

1.  **Initialization (All Processes):**
    *   MPI environment is initialized (`MPI.COMM_WORLD`).
    *   Each process gets its rank and the total size (number of processes).

2.  **Setup (Root Process):**
    *   Fits the LabelEncoder on `synthetic_test_data.csv` and saves it.
    *   Loads the pre-trained ML model (`fraud_rf_model.pkl`).
    *   Initializes the mock transaction queue by reading all data from `synthetic_test_data.csv` into an in-memory list and creating an iterator over it.

3.  **Broadcast**
    *   The root process broadcasts the loaded model and the fitted LabelEncoder instance to the other worker processes.

4.  **Main Processing (All Processes):**
    *   The application enters a loop to process transactions in batches.
    *   **Data Batching and Preprocessing (Root):**
        *   The root calls mock_pull_from_queue() to get a batch of transactions (up to `P`, the number of MPI processes).
        *   If the mock queue is empty or a configured max number of batches is reached, the root prepares a shutdown signal.
        *   Otherwise, it preprocesses each raw transaction in the batch into a numerical feature array suitable for the model.
    *   **Task Distribution (`comm.scatter`):**
        *   The root scatters the list of preprocessed feature arrays (or shutdown signals) to all processes. Each process receives one item.
    *   **Parallel Prediction (All Processes):**
        *   Each process checks if it received a shutdown signal. If so, it prepares to terminate.
        *   If it received valid feature data, it uses its local model to make a prediction.
    *   **Result Collection (`comm.gather`):**
        *   All processes send their prediction (or shutdown/error signal) back to the root process.
    *   **Result Handling (Root):**
        *   The root process collects the gathered results.
        *   If all results are shutdown signals, the root initiates service termination.
        *   For valid predictions, it associates them with original transaction details and "pushes" them to the mock_results_lst using mock_push_to_results_queue(). It also stores them for final CSV output.
    *   The loop continues until a shutdown condition is met.