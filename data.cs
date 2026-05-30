static class SampleDatasets
{
    public static (double[][] x, double[] y) Linear = (
        new double[][]
        {
            [1, 1, 2],
            [1, 2, 3],
            [1, 3, 4],
            [1, 4, 5],
            [1, 5, 6]
        },
        new double[] { 3, 5, 7, 9, 11 }
    );
    
    public static (double[][] x, double[] y) Logistic = (
        new double[][]
        {
            [1, 1, 2],
            [1, 2, 3],
            [1, 3, 4],
            [1, 4, 5],
            [1, 5, 6]
        },
        new double[] { 0, 0, 0, 1, 1 }
    );
}