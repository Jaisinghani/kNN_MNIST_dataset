#running knn for optimal K
predictor = MNISTPredictor(copyOfTrainingDataSet, optimal_k)
optimal_k_predictions = predict_test_set(predictor, testImagesSet)
optimal_k_accuracy = evaluate_prediction(optimal_k_predictions, testLabelsSet)