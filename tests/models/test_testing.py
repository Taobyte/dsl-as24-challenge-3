from src.train_validate import compute_metrics

def test_predictions_len():

    df = compute_metrics(10)

    assert len(df) == 10


def test_df_columns():

    df = compute_metrics(10)

    assert set(df.columns) == set(['cc_mean', 'cc_std', 'max_amp_diff_mean', 'max_amp_diff_std', 'p_wave_mean', 'p_wave_std', 'snr'])