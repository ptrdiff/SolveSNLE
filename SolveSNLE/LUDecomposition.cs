using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SolveSNLE
{
    class LUDecomposition
    {
        private static Matrix<double>[] Decomposition(Matrix<double> originMatrix)
        {
            void swapRow(ref Matrix<double> a, int i, int j)
            {
                var first = a.Row(i);
                var second = a.Row(j);
                a.SetRow(i, second);
                a.SetRow(j, first);

            }
            void swapColumn(ref Matrix<double> a, int i, int j)
            {
                var first = a.Column(i);
                var second = a.Column(j);
                a.SetColumn(i, second);
                a.SetColumn(j, first);

            }

            Matrix<double> C = Matrix.Build.DenseOfMatrix(originMatrix);
            Matrix<double> P = Matrix.Build.DenseIdentity(C.RowCount, C.ColumnCount);
            Matrix<double> Q = Matrix.Build.DenseIdentity(C.RowCount, C.ColumnCount);

            int maxElRow = -1;
            int maxElCol = -1;

            for (int i = 0; i < C.RowCount; ++i)
            {

                double pivotValue = 0;
                for (int row = i; row < C.RowCount; ++row)
                {
                    for (int col = i; col < C.ColumnCount; ++col)
                    {
                        if (Math.Abs(C[row, col]) > pivotValue)
                        {
                            pivotValue = Math.Abs(C[row, col]);
                            maxElRow = row;
                            maxElCol = col;
                        }
                    }
                }
                if (pivotValue < 10e-16)
                {
                    continue;
                }

                swapRow(ref C,i,maxElRow);
                swapColumn(ref C, i, maxElCol);
                swapRow(ref P, i, maxElRow);
                swapColumn(ref Q, i, maxElCol);

                for (int row = i + 1; row < C.RowCount; ++row)
                {
                    C[row, i] /= C[i, i];
                    for (int col = i + 1; col < C.ColumnCount; ++col)
                    {
                        C[row, col] -= C[row, i] * C[i, col];
                    }
                }
            }
            Matrix<double> lower = Matrix.Build.DenseIdentity(C.RowCount, C.ColumnCount);
            Matrix<double> upper = Matrix.Build.Dense(C.RowCount, C.ColumnCount);

            for (int row = 0; row < C.RowCount; ++row)
            {
                for (int col = 0; col < C.ColumnCount; ++col)
                {
                    if (row > col)
                    {
                        lower[row, col] = C[row, col];
                    }
                    else
                    {
                        upper[row, col] = C[row, col];
                    }
                }
            }
            return new Matrix<double>[] { P, lower, upper, Q} ;
        }
        public static Vector<double> SolveSLE(Matrix<double> originMatrix, Vector<double> b)
        {
            var PLUQ = Decomposition(originMatrix);
            Vector<double> y = Vector.Build.Dense(originMatrix.RowCount);
            b = PLUQ[0] * b;
            y[0] = b[0];

            for (int i = 1; i < originMatrix.RowCount; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < i; ++j)
                {
                    sum += y[j] * PLUQ[1][i, j];
                }
                y[i] = b[i] - sum;
            }

            Vector<double> x = Vector.Build.Dense(originMatrix.RowCount);

            x[originMatrix.RowCount - 1] = y[originMatrix.RowCount - 1] / PLUQ[2][originMatrix.RowCount - 1, originMatrix.ColumnCount - 1];

            for (int i = 2; i < originMatrix.RowCount + 1; ++i)
            {
                double sum = 0.0;
                for (int j = 1; j < i; ++j)
                {
                    sum += x[originMatrix.RowCount - j] * PLUQ[2][originMatrix.RowCount - i, originMatrix.ColumnCount - j];
                }
                x[originMatrix.RowCount - i] = (y[originMatrix.RowCount - i] - sum) / PLUQ[2][originMatrix.RowCount - i, originMatrix.RowCount - i];
            }

            return PLUQ[3] * x;
        }
        private static Matrix<double> Inverse(Matrix<double> originMatrix)
        {
            Matrix<double> inv = Matrix.Build.Dense(originMatrix.RowCount, originMatrix.ColumnCount);
            for (int i = 0; i < originMatrix.RowCount; ++i)
            {
                Vector<double> z = Vector.Build.Dense(originMatrix.RowCount);
                z[i] = 1;
                Vector<double> s = SolveSLE(originMatrix, z);
                for (int j = 0; j < originMatrix.RowCount; ++j)
                {
                    inv[j, i] = s[j];
                }
            }
            return inv;
        }
    }
}
