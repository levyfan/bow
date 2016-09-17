package com.github.levyfan.reid.pca;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.jblas.DoubleMatrix;
import org.jblas.Singular;

import java.util.Arrays;
import java.util.List;

/**
 * @author fanliwen
 */
public class BlasPca {

    public double[] mean;

    public double[] eigvalue;

    public DoubleMatrix u;

    public DoubleMatrix ux;

    public BlasPca(DoubleMatrix u, double[] mean) {
        this.u = u;
        this.mean = mean;
    }

    public BlasPca(List<double[]> hists) {
        DoubleMatrix matrix = new DoubleMatrix(hists.toArray(new double[0][])).transpose();

        this.mean = new double[matrix.getRows()];
        for (int row = 0; row < matrix.getRows(); row++) {
            this.mean[row] = new Mean().evaluate(matrix.getRow(row).toArray());
            for (int col = 0; col < matrix.getColumns(); col++) {
                matrix.put(row, col, matrix.get(row, col) - this.mean[row]);
            }
        }

        System.out.println("start pca");
        DoubleMatrix[] svd = Singular.sparseSVD(matrix);
        System.out.println("done");

        this.ux = svd[0].transpose().mmul(matrix);
        this.u = svd[0];

        DoubleMatrix s = svd[1];
        this.eigvalue = s.toArray();
        for (int i = 0; i < this.eigvalue.length; i++) {
            this.eigvalue[i] = this.eigvalue[i] * this.eigvalue[i];
        }
    }

    public DoubleMatrix apply(List<double[]> hists) {
        DoubleMatrix matrix = new DoubleMatrix(hists.toArray(new double[0][])).transpose();
        for (int row = 0; row < matrix.getRows(); row++) {
            for (int col = 0; col < matrix.getColumns(); col++) {
                matrix.put(row, col, matrix.get(row, col) - this.mean[row]);
            }
        }

        return this.u.transpose().mmul(matrix);
    }

    public double[] apply(double[] hist) {
        DoubleMatrix matrix = new DoubleMatrix(Arrays.copyOf(hist, hist.length));
        for (int row = 0; row < matrix.getRows(); row ++) {
            matrix.put(row, 0, matrix.get(row, 0) - this.mean[row]);
        }

        return this.u.transpose().mmul(matrix).toArray();
    }
}
