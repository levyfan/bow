package com.github.levyfan.reid.viper;

import com.github.levyfan.reid.pca.BlasPca;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author fanliwen
 */
public class Viper {

    private RealMatrix select;

    public Viper() throws URISyntaxException, IOException {
        File file = new File(this.getClass().getResource("/randselect10.mat").toURI());
        MatFileReader reader = new MatFileReader(file);
        MLDouble mlDouble = (MLDouble) reader.getMLArray("selectsample");
        this.select = MatrixUtils.createRealMatrix(mlDouble.getArray());
    }

    public double[] eval(List<double[]> histA, List<double[]> histB) {
        int lookrank = select.getRowDimension();
        int nloop = select.getColumnDimension();
        RealMatrix accuracy = MatrixUtils.createRealMatrix(lookrank, nloop);

        for (int loop = 0; loop < nloop; loop ++) {
            double[] s = select.getColumn(loop);

            List<Integer> testIndex = new ArrayList<>();
            for (int i = 0; i < s.length; i++) {
                testIndex.add((int) (s[i] - 1));
            }

            List<double[]> testHistA = new ArrayList<>();
            List<double[]> trainHistA = new ArrayList<>();
            for (int i = 0; i < histA.size(); i++) {
                if (testIndex.contains(i)) {
                    testHistA.add(histA.get(i));
                } else {
                    trainHistA.add(histA.get(i));
                }
            }

            List<double[]> testHistB = new ArrayList<>();
            List<double[]> trainHistB = new ArrayList<>();
            for (int i = 0; i < histB.size(); i++) {
                if (testIndex.contains(i)) {
                    testHistB.add(histB.get(i));
                } else {
                    trainHistB.add(histB.get(i));
                }
            }

            List<double[]> trainHist = Lists.newArrayList(Iterables.concat(trainHistA, trainHistB));
            BlasPca pca = new BlasPca(trainHist);

            List<double[]> testHist = Lists.newArrayList(Iterables.concat(testHistA, testHistB));
            DoubleMatrix matrix = new DoubleMatrix(testHist.toArray(new double[0][])).transpose();
            for (int row = 0; row < matrix.getRows(); row++) {
                for (int col = 0; col < matrix.getColumns(); col++) {
                    matrix.put(row, col, matrix.get(row, col) - pca.mean[row]);
                }
            }

            DoubleMatrix ux = pca.u.transpose().mmul(matrix);
            DoubleMatrix histPcaA = ux.get(
                    new IntervalRange(0, ux.getRows()), new IntervalRange(0, testHistA.size()));
            DoubleMatrix histPcaB = ux.get(
                    new IntervalRange(0, ux.getRows()), new IntervalRange(testHistA.size(), testHist.size()));
            RealMatrix testscore = MatrixUtils.createRealMatrix(
                    histPcaA.transpose().mmul(histPcaB).toArray2());

            double[] countq = new double[lookrank];
            for (int i = 0; i < testIndex.size(); i++) {
                double[] rowScores = testscore.getRow(i);
                int rowIndex = 0;
                for (double rowScore : rowScores) {
                    if (rowScore > rowScores[i]) {
                        rowIndex++;
                    }
                }

                double[] colScores = testscore.getColumn(i);
                int colIndex = 0;
                for (double colScore : colScores) {
                    if (colScore > colScores[i]) {
                        colIndex++;
                    }
                }

                for (int ii = rowIndex; ii < countq.length; ii++) {
                    countq[ii] ++;
                }
                for (int ii = colIndex; ii < countq.length; ii++) {
                    countq[ii] ++;
                }
            }
            accuracy.setColumn(loop, countq);
        }

        double[] MR = new double[lookrank];
        for (int row = 0; row < accuracy.getRowDimension(); row++) {
            MR[row] = new Mean().evaluate(accuracy.getRow(row))/lookrank/2;
        }
        return MR;
    }

    public double[] eval(RealMatrix score) {
        int lookrank = select.getRowDimension();

        int nloop = select.getColumnDimension();
        RealMatrix accuracy = MatrixUtils.createRealMatrix(lookrank, nloop);
        for (int loop = 0; loop < nloop; loop++) {
            double[] s = select.getColumn(loop);

            int[] ss = new int[s.length];
            for (int i = 0; i < s.length; i++) {
                ss[i] = (int) (s[i] - 1);
            }
            RealMatrix testscore = score.getSubMatrix(ss, ss);

            double[] countq = new double[lookrank];
            for (int i = 0; i < ss.length; i++) {
                double[] rowScores = testscore.getRow(i);
                int rowIndex = 0;
                for (double rowScore : rowScores) {
                    if (rowScore > rowScores[i]) {
                        rowIndex++;
                    }
                }

                double[] colScores = testscore.getColumn(i);
                int colIndex = 0;
                for (double colScore : colScores) {
                    if (colScore > colScores[i]) {
                        colIndex++;
                    }
                }

                for (int ii = rowIndex; ii < countq.length; ii++) {
                    countq[ii] ++;
                }
                for (int ii = colIndex; ii < countq.length; ii++) {
                    countq[ii] ++;
                }
            }

            accuracy.setColumn(loop, countq);
        }

        double[] MR = new double[lookrank];
        for (int row = 0; row < accuracy.getRowDimension(); row++) {
            MR[row] = new Mean().evaluate(accuracy.getRow(row))/lookrank/2;
        }
        return MR;
    }
}
