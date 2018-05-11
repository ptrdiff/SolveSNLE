using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SolveSNLE
{
    class Program
    {
        static void Main(string[] args)
        {
            First();
            Second();
        }

        private static void Second()
        {
            Vector<double> sX = Vector.Build.DenseOfArray(new double[] { 0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5 });
            Console.WriteLine(SnleSolver.Newton(sX,1e-15));
            Console.WriteLine(SnleSolver.NewtonMod(sX, 1e-15));
            Console.WriteLine(SnleSolver.NewtonMix(sX, 1e-15, 3));
            Console.WriteLine(SnleSolver.NewtonHybrid(sX, 1e-15, 3));
            SnleSolver.Diagnostic(sX, 1e-15, 3, 3);
        }

        private static void First()
        {
            double fun(double x) => x * Math.Log(x + 1) - 0.3;
            Console.WriteLine(NleSolver.SolveNewton(fun, -0.9, 1, 1e-4));
        }
    }
}
