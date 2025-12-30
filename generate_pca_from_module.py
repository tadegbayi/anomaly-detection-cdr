from run_unsupervised_compare import load_sample, prepare_features, run_all, summarize_and_plot

if __name__ == '__main__':
    df = load_sample()
    Xs, Xraw, scaler = prepare_features(df)
    results = run_all(Xs)
    summarize_and_plot(df, Xs, results)
    print('Done generate_pca_from_module')
