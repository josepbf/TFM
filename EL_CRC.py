import pandas as pd

def CRC_lambda_threshold_finder(calibration_data, alpha = 0.1)
    """
    recall 0.9
    fnr 0.1
    alpha 0.1
    fnr_ajustat = 0.091
    recall_ajustat = 0.909 -> perf pq queda per sobre del valor de recall minim fixat per alpha"""
    """
    threshold puja 1 0.8 0.6
    recall baixa   0 0.6 0.8
    fnr puja       1 0.4 0.2
    volen fnr baix
    - volen el primer threhold en el que fnr =< 0.091
    - més petit vol dir que començem per threhold 1 i anem baixant fins 
    que la fnr empirica sigui més petit o igual 0.091."""
    B = 1
    num_samples = len(calibration_data)
    lambda_threshold
    
    FNR_max
    FNR = 1 - recall
    
    alpha
    return lambda_threshold


def forecast_safety_powerloss(output):
    # Forecasting performance loss according to the type of defect detected. 
    # Quantify the performance of solar cells. Still no clear relationship between 
    # the defect problems reflected in EL images and the actual performance.

    table = {
        "Type of defect": ["Finger Failure A", "Crack A/B", "Crack C" "Finger Failure B"]
        "Defect index": [1, 2, 3, -1]
        "Safety": ["A", "B", "B(f)", "B(f)"]
        "Power loss": ["A", "C", "C", "C"]
    }

    # Finger Failure B (when finder failure overlapp Crack)