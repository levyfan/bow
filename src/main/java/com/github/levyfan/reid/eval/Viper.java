package com.github.levyfan.reid.eval;

import com.github.levyfan.reid.pca.BlasPca;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileType;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.util.Pair;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.util.List;

/**
 * @author fanliwen
 */
public class Viper {

    private RealMatrix select;

    public Viper() throws URISyntaxException, IOException {
        try (InputStream stream = this.getClass().getResourceAsStream("/randselect10.mat")) {
            MatFileReader reader = new MatFileReader(stream, MatFileType.Regular);
            MLDouble mlDouble = (MLDouble) reader.getMLArray("selectsample");
            this.select = MatrixUtils.createRealMatrix(mlDouble.getArray());
        }
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

    public Pair<double[], RealMatrix> eval(List<double[]> histA, List<double[]> histB, boolean enablePCA) {
        RealMatrix score;
        if (enablePCA) {
            List<double[]> hist = Lists.newArrayList(Iterables.concat(histA, histB));
            BlasPca pca = new BlasPca(hist);

//            DoubleMatrix u = pca.u;
//            DoubleMatrix histPcaA = u.transpose().mmul(
//                    new DoubleMatrix(histA.toArray(new double[0][])).transpose());
//            DoubleMatrix histPcaB = u.transpose().mmul(
//                    new DoubleMatrix(histB.toArray(new double[0][])).transpose());
            DoubleMatrix histPcaA = pca.ux.get(
                    new IntervalRange(0, pca.ux.getRows()), new IntervalRange(0, histA.size()));
            DoubleMatrix histPcaB = pca.ux.get(
                    new IntervalRange(0, pca.ux.getRows()), new IntervalRange(histA.size(), hist.size()));
            score = MatrixUtils.createRealMatrix(
                    histPcaA.transpose().mmul(histPcaB).toArray2());
        } else {
            score = MatrixUtils.createRealMatrix(histA.toArray(new double[0][]))
                    .multiply(MatrixUtils.createRealMatrix(histB.toArray(new double[0][])).transpose());
        }

        return Pair.create(eval(score), score);
    }
}
