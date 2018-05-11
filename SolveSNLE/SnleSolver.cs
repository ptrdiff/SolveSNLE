using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Diagnostics;

namespace SolveSNLE
{
    class SnleSolver
    {
        static Func<Vector<double>, Vector<double>> SLExMod = (Vector<double> x) =>
        LUDecomposition.SolveSLE(CalculateJacobiMatrixValue(x), -CalculateFuncValue(x));

        static Func<Vector<double>, double>[] functions =
        {
            (Vector<double> x) => Math.Cos(x[0]*x[1]) - Math.Exp(-3*x[2]) + x[3] * Math.Pow(x[4],2) - x[5] - Math.Sinh(2*x[7]) * x[8] + 2 * x[9] + 2.0004339741653854440,
            (Vector<double> x) => Math.Sin(x[0]*x[1]) + x[2]*x[8]*x[6] - Math.Exp(-x[9]+x[5]) + 3*Math.Pow(x[4],2) - x[5] * (x[7] + 1) + 10.886272036407019994,
            (Vector<double> x) => x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] - x[9] - 3.1361904761904761904,
            (Vector<double> x) => 2 * Math.Cos(-x[8] + x[3]) + x[4] / (x[2] + x[0]) - Math.Sin(Math.Pow(x[1], 2)) + Math.Pow(Math.Cos(x[6]*x[9]),2) - x[7] - 0.1707472705022304757,
            (Vector<double> x) => Math.Sin(x[4]) + 2*x[7]*(x[2] + x[0]) - Math.Exp(-x[6]*(-x[9]+x[5])) + 2 * Math.Cos(x[1]) - 1 / (x[3] - x[8]) -  0.3685896273101277862,
            (Vector<double> x) => Math.Exp(x[0] - x[3] - x[8]) + Math.Pow(x[4],2)/x[7] + Math.Cos(3*x[9]*x[1])/2 - x[5]*x[2] + 2.049108601677187511,
            (Vector<double> x) => Math.Pow(x[1],3)*x[6] - Math.Sin(x[9]/x[4] + x[7]) + (x[0]-x[5])*Math.Cos(x[3]) + x[2] - 0.7380430076202798014,
            (Vector<double> x) => x[4]*Math.Pow((x[0] - 2*x[5]),2) - 2 * Math.Sin(-x[8]+x[2]) + 1.5*x[3] - Math.Exp(x[1]*x[6]+x[9]) + 3.566832198969380904,
            (Vector<double> x) => 7/x[5] + Math.Exp(x[4]+x[3]) - 2*x[1]*x[7]*x[9]*x[6] + 3*x[8] - 3*x[0] - 8.439473450838325749,
            (Vector<double> x) => x[9]*x[0] + x[8]*x[1] - x[7]*x[2] + Math.Sin(x[3]+x[4]+x[5])*x[6] - 0.7823809523809523809
        };
        static Func<Vector<double>, double>[,] jacobiMatrix =
        {
            {
              (Vector<double> x) => -Math.Sin(x[0] * x[1]) * x[1],
              (Vector<double> x) => -Math.Sin(x[0] * x[1]) * x[0],
              (Vector<double> x) => 3.0 * Math.Exp(-(3.0 * x[2])),
              (Vector<double> x) => Math.Pow(x[4],2),
              (Vector<double> x) => 2.0 * x[3] * x[4],
              (Vector<double> x) => -1,
              (Vector<double> x) => 0,
              (Vector<double> x) => -2.0 * Math.Cosh(2.0 * x[7]) * x[8],
              (Vector<double> x) => -Math.Sinh(2.0 * x[7]),
              (Vector<double> x) => 2
            },
            {
              (Vector<double> x) => Math.Cos(x[0] * x[1]) * x[1],
              (Vector<double> x) => Math.Cos(x[0] * x[1]) * x[0],
              (Vector<double> x) => x[8] * x[6],
              (Vector<double> x) => 0,
              (Vector<double> x) => 6 * x[4],
              (Vector<double> x) => -Math.Exp(-x[9] + x[5]) - x[7] - 0.1e1,
              (Vector<double> x) => x[2] * x[8],
              (Vector<double> x) => -x[5],
              (Vector<double> x) => x[2] * x[6],
              (Vector<double> x) => Math.Exp(-x[9] + x[5])
            },
            {
              (Vector<double> x) => 1,
              (Vector<double> x) => -1,
              (Vector<double> x) => 1,
              (Vector<double> x) => -1,
              (Vector<double> x) => 1,
              (Vector<double> x) => -1,
              (Vector<double> x) => 1,
              (Vector<double> x) => -1,
              (Vector<double> x) => 1,
              (Vector<double> x) => -1
            },
            {
              (Vector<double> x) => - x[4] * Math.Pow(x[2] + x[0], -2),
              (Vector<double> x) => -2.0 * Math.Cos(x[1] * x[1]) * x[1],
              (Vector<double> x) => - x[4] * Math.Pow( x[2] + x[0], -2),
              (Vector<double> x) => -2.0 * Math.Sin(-x[8] + x[3]),
              (Vector<double> x) => 1.0 / ( x[2] + x[0]),
              (Vector<double> x) => 0.0,
              (Vector<double> x) => -2.0 * Math.Cos(x[6] * x[9]) * Math.Sin(x[6] * x[9]) * x[9],
              (Vector<double> x) => -1.0,
              (Vector<double> x) => 2.0 * Math.Sin(-x[8] + x[3]),
              (Vector<double> x) => -2.0 * Math.Cos(x[6] * x[9]) * Math.Sin(x[6] * x[9]) * x[6]
            },
            {
              (Vector<double> x) => 2.0 * x[7],
              (Vector<double> x) => -2.0 * Math.Sin(x[1]),
              (Vector<double> x) => 2.0 * x[7],
              (Vector<double> x) => Math.Pow(-x[8] + x[3], -2.0),
              (Vector<double> x) => Math.Cos(x[4]),
              (Vector<double> x) => x[6] * Math.Exp(-x[6] * (-x[9] + x[5])),
              (Vector<double> x) => -(x[9] - x[5]) * Math.Exp(-x[6]* (-x[9] + x[5])),
              (Vector<double> x) => (2.0 * x[2]) + 2.0 * x[0],
              (Vector<double> x) => -Math.Pow(-x[8] + x[3], -2.0),
              (Vector<double> x) => -x[6] * Math.Exp(-x[6]* (-x[9] + x[5]))
            },
            {
              (Vector<double> x) => Math.Exp(x[0] - x[3] - x[8]),
              (Vector<double> x) => -3.0 / 2.0 * Math.Sin(3.0 * x[9] * x[1]) * x[9],
              (Vector<double> x) => -x[5],
              (Vector<double> x) => -Math.Exp(x[0] - x[3] - x[8]),
              (Vector<double> x) => 2.0 * x[4] / x[7],
              (Vector<double> x) => -x[2],
              (Vector<double> x) => 0,
              (Vector<double> x) => -x[4] * x[4] * Math.Pow(x[7], -2),
              (Vector<double> x) => -Math.Exp(x[0] - x[3] - x[8]),
              (Vector<double> x) => -3.0 / 2.0 * Math.Sin(3.0 * x[9] * x[1]) * x[1]
            },
            {
              (Vector<double> x) => Math.Cos(x[3]),
              (Vector<double> x) => 3.0 * x[1] * x[1] * x[6],
              (Vector<double> x) => 1,
              (Vector<double> x) => -(x[0] - x[5]) * Math.Sin(x[3]),
              (Vector<double> x) => Math.Cos(x[9] / x[4] + x[7]) * x[9] * Math.Pow(x[4], -2),
              (Vector<double> x) => -Math.Cos(x[3]),
              (Vector<double> x) => Math.Pow(x[1], 3.0),
              (Vector<double> x) => -Math.Cos(x[9] / x[4] + x[7]),
              (Vector<double> x) => 0,
              (Vector<double> x) => -Math.Cos(x[9] / x[4] + x[7]) / x[4]
            },
            {
              (Vector<double> x) => 2.0 * x[4] * (x[0] - 2.0 * x[5]),
              (Vector<double> x) => -x[6] * Math.Exp(x[1] * x[6] + x[9]),
              (Vector<double> x) => -2.0 * Math.Cos(-x[8] + x[2]),
              (Vector<double> x) => 0.15e1,
              (Vector<double> x) => Math.Pow(x[0] - 2.0 * x[5], 2.0),
              (Vector<double> x) => -4.0 * x[4] * (x[0] - 2.0 * x[5]),
              (Vector<double> x) => -x[1] * Math.Exp(x[1] * x[6] + x[9]),
              (Vector<double> x) => 0.0,
              (Vector<double> x) => 2.0 * Math.Cos(-x[8] + x[2]),
              (Vector<double> x) => -Math.Exp(x[1] * x[6] + x[9])
            },
            {
              (Vector<double> x) => -3.0,
              (Vector<double> x) => -2.0 * x[7] * x[9] * x[6],
              (Vector<double> x) => 0.0,
              (Vector<double> x) => Math.Exp(x[4] + x[3]),
              (Vector<double> x) => Math.Exp(x[4] + x[3]),
              (Vector<double> x) => -0.7e1 * Math.Pow(x[5], -2.0),
              (Vector<double> x) => -2.0 * x[1] * x[7] * x[9],
              (Vector<double> x) => -2.0 * x[1] * x[9] * x[6],
              (Vector<double> x) => 3.0,
              (Vector<double> x) => -2.0 * x[1] * x[7] * x[6]
            },
            {
              (Vector<double> x) => x[9],
              (Vector<double> x) => x[8],
              (Vector<double> x) => -x[7],
              (Vector<double> x) => Math.Cos(x[3] + x[4] + x[5]) * x[6],
              (Vector<double> x) => Math.Cos(x[3] + x[4] + x[5]) * x[6],
              (Vector<double> x) => Math.Cos(x[3] + x[4] + x[5]) * x[6],
              (Vector<double> x) => Math.Sin(x[3] + x[4] + x[5]),
              (Vector<double> x) => -x[2],
              (Vector<double> x) => x[1],
              (Vector<double> x) => x[0]
            }
        };

        static Vector<double> CalculateFuncValue(Vector<double> x)
        {
            Vector<double> value = CreateVector.Dense<double>(x.Count);
            for(int i = 0; i < functions.Length; ++i)
            {
                value[i] = functions[i](x);
            }
            return value;
        }
        static Matrix<double> CalculateJacobiMatrixValue(Vector<double> x)
        {
            Matrix<double> value = new DenseMatrix(x.Count,x.Count);

            for (int i = 0; i < functions.Length; ++i)
            {
                for (int j = 0; j < functions.Length; ++j)
                {
                    value[i,j] = jacobiMatrix[i,j](x);
                }
            }
            return value;
        }

        public static Tuple<Vector<double>, int> Newton(Vector<double> x, double eps)
        {
            Vector<double> y = CreateVector.Dense<double>(x.Count);
            int count = 0;
            do
            {
                y = x;
                x = x + LUDecomposition.SolveSLE(CalculateJacobiMatrixValue(x), -CalculateFuncValue(x));
                ++count;
            } while ((x - y).L2Norm() >= eps);
            return Tuple.Create(x, count);
        }

        public static Tuple<Vector<double>, int> NewtonMod(Vector<double> x, double eps)
        {
            Vector<double> y = CreateVector.Dense<double>(x.Count);
            var tmpJacobi = CalculateJacobiMatrixValue(x);
            int count = 0;
            do
            {
                y = x;
                x = x + LUDecomposition.SolveSLE(tmpJacobi, -CalculateFuncValue(x));
                ++count;
            } while ((x - y).L2Norm() >= eps);
            return Tuple.Create(x, count);
        }

        public static Tuple<Vector<double>,int> NewtonMix(Vector<double> x, double eps, int k)
        {
            Vector<double> y = CreateVector.Dense<double>(x.Count);
            int count = 0;
            var tmpJacobi = Matrix.Build.Dense(x.Count, x.Count);
            do
            {
                y = x;
                if (count < k)
                {
                    tmpJacobi = CalculateJacobiMatrixValue(x);
                }
                x = x + LUDecomposition.SolveSLE(tmpJacobi, -CalculateFuncValue(x));
                ++count;
            } while ((x - y).L2Norm() >= eps && count < 100);
            return Tuple.Create(x, count);
        }

        public static Tuple<Vector<double>, int> NewtonHybrid(Vector<double> x, double eps, int k)
        {
            Vector<double> y = CreateVector.Dense<double>(x.Count);
            int count = 0;
            var tmpJacobi = CalculateJacobiMatrixValue(x);
            do
            {
                ++count;
                y = x;
                if (count % k == 0)
                {
                    tmpJacobi = CalculateJacobiMatrixValue(x);
                }
                x = x + LUDecomposition.SolveSLE(tmpJacobi, -CalculateFuncValue(x));
            } while ((x - y).L2Norm() >= eps && count < 100);
            return Tuple.Create(x, count);
        }

        public static void Diagnostic(Vector<double> x, double eps, int k, int j)
        {
            Newton(x, eps);
            NewtonMod(x, eps);
            NewtonMix(x, eps, k);
            NewtonHybrid(x, eps, j);

            Stopwatch stopwatch = new Stopwatch();

            Console.WriteLine("-----Newton-----");
            stopwatch.Start();
            for (var i = 0; i < 100; ++i)
            {
                Newton(x, eps);
            }
            stopwatch.Stop();
            Console.WriteLine(stopwatch.ElapsedTicks/100);

            Console.WriteLine("-----NewtonMod-----");
            stopwatch.Restart();
            for (var i = 0; i < 100; ++i)
            {
                NewtonMod(x, eps);
            }
            stopwatch.Stop();
            Console.WriteLine(stopwatch.ElapsedTicks/100);

            Console.WriteLine("-----NewtonMix-----");
            stopwatch.Restart();
            for (var i = 0; i < 100; ++i)
            {
                NewtonMix(x, eps, k);
            }
            stopwatch.Stop();
            Console.WriteLine(stopwatch.ElapsedTicks/100);

            Console.WriteLine("-----NewtonHydrid-----");
            stopwatch.Restart();
            for (var i = 0; i < 100; ++i)
            {
                NewtonHybrid(x, eps, j);
            }
            stopwatch.Stop();
            Console.WriteLine(stopwatch.ElapsedTicks / 100);
        }
    }
}
