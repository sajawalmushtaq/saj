using System;
using System.Formats.Asn1;
using System.Globalization;
using System.IO;
using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;

public class Reviews
{
    [LoadColumn(0)]
    public bool Sent { get; set; }

    [LoadColumn(1)]
    public string Title { get; set; }

    [LoadColumn(2)]
    public string Review { get; set; }
}

public class ReviewPrediction
{
    public bool PredictedLabel { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}

public class Program
{
    private const string ModelPath = "model.zip";

    public static void Main()
    {
        string testReviewsFilePath = Path.Combine(Environment.CurrentDirectory, "Data", "test-reviews.csv");
        var mlContext = new MLContext();

        var model = File.Exists(ModelPath)
            ? mlContext.Model.Load(ModelPath, out _)
            : TrainModel(mlContext);

        var predictionEngine = mlContext.Model.CreatePredictionEngine<Reviews, ReviewPrediction>(model);
        var reviews = ReadReviewsFromCsv(testReviewsFilePath);

        foreach (var review in reviews)
        {
            var prediction = predictionEngine.Predict(review);
            Console.WriteLine($"Review: {review.Review}\nPredicted Sentiment: {(prediction.PredictedLabel ? "Positive" : "Negative")}\nProbability: {prediction.Probability:F2}\nScore: {prediction.Score:F2}\n");
        }
    }

    private static ITransformer TrainModel(MLContext mlContext)
    {
        string trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train-reviews-micro.csv");
        var data = mlContext.Data.LoadFromTextFile<Reviews>(trainingDataPath, separatorChar: ',', hasHeader: true);

        // Define the pipeline for text classification
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(Reviews.Review))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(Reviews.Sent), featureColumnName: "Features"));

        // Fit the model
        var model = pipeline.Fit(data);

        // Save the trained model
        mlContext.Model.Save(model, data.Schema, ModelPath);

        return model;
    }


    private static Reviews[] ReadReviewsFromCsv(string filePath)
    {
        using var reader = new StreamReader(filePath);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        var reviewsList = new System.Collections.ArrayList();

        while (csv.Read())
        {
            var review = csv.GetRecord<Reviews>();
            reviewsList.Add(review);
        }

        return (Reviews[])reviewsList.ToArray(typeof(Reviews));
    }
}
