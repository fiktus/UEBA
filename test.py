import pandas as pd
from models_back import AutModel
from models import UEBASequenceAutoencoder

if __name__ == "__main__":
    df = pd.read_csv("lastadatause.csv", low_memory=False)

    ueba = UEBASequenceAutoencoder(window_size=40, stride=5)
    results = ueba.train(df)
    print(results['is_anomaly'].value_counts())

    ueba.save("ueba_model9.pt")


    loaded_ueba = UEBASequenceAutoencoder()
    loaded_ueba.load_model("ueba_model9.pt")
    df = pd.read_csv('test.csv', on_bad_lines='skip')
    test_df = df

    mse_scores, anomaly_flags, status = loaded_ueba.predict(test_df)

    for i in range(len(test_df)):
        print(f"строка {i}    MSE: {mse_scores[i]:.5f}     статус: {status[i]}")
