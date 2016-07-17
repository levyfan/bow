package com.github.levyfan.reid.pca;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

import java.util.List;

/**
 * @author fanliwen
 */
public class Pca {

    public double[] mean;

    public RealMatrix ux;

    public RealMatrix u;

    public double[] eigvalue;

    public Pca(List<double[]> hists) {
        RealMatrix matrix = MatrixUtils
                .createRealMatrix(hists.toArray(new double[0][]))
                .transpose();

        this.mean = new double[matrix.getRowDimension()];
        for (int row = 0; row < matrix.getRowDimension(); row++) {
            this.mean[row] = new Mean().evaluate(matrix.getRow(row));
            for (int col = 0; col < matrix.getColumnDimension(); col++) {
                matrix.setEntry(row, col, matrix.getEntry(row, col) - this.mean[row]);
            }
        }

        System.out.println("start pca");
        SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
        System.out.println("done");

        this.ux = svd.getUT().multiply(matrix);
        this.u = svd.getU();

        this.eigvalue = svd.getSingularValues();
        for (int i = 0; i < this.eigvalue.length; i++) {
            this.eigvalue[i] = this.eigvalue[i] * this.eigvalue[i];
        }
    }
}
