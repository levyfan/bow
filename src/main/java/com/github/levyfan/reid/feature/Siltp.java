package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.sp.SuperPixel;
import com.google.common.primitives.Doubles;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.summary.Sum;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * @author fanliwen
 */
class Siltp implements Feature {

    private static final double tau = 0.3;
    private static final int total = 81;

    private static RealMatrix siltp(RealMatrix gray, double tau, int R) {
        int h = gray.getRowDimension();
        int w = gray.getColumnDimension();

        RealMatrix i0 = MatrixUtils.createRealMatrix(h+2*R, w+2*R);
        i0.setSubMatrix(gray.getData(), R, R);

        double[] rowR = i0.getRow(R);
        for (int i = 0; i < R; i++) {
            i0.setRow(i, rowR);
        }

        double[] rowEndR = i0.getRow(h+R-1);
        for (int i = h+R; i < h+2*R; i++) {
            i0.setRow(i, rowEndR);
        }

        double[] colR = i0.getColumn(R);
        for (int i = 0; i < R; i++) {
            i0.setColumn(i, colR);
        }

        double[] colEndR = i0.getColumn(w+R-1);
        for (int i = w+R; i < w+2*R; i++) {
            i0.setColumn(i, colEndR);
        }

        RealMatrix i1 = i0.getSubMatrix(R, h+R-1, 2*R, w+2*R-1);
        RealMatrix i3 = i0.getSubMatrix(0, h-1, R, w+R-1);
        RealMatrix i5 = i0.getSubMatrix(R, h+R-1, 0, w-1);
        RealMatrix i7 = i0.getSubMatrix(2*R, h+2*R-1, R, w+R-1);

        RealMatrix l = gray.scalarMultiply(1-tau);
        RealMatrix u = gray.scalarMultiply(1+tau);

        RealMatrix matrix = MatrixUtils.createRealMatrix(h, w);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int v1 = (i1.getEntry(i, j)<l.getEntry(i, j) ? 1 : 0) + (i1.getEntry(i, j)>u.getEntry(i, j) ? 2 : 0);
                int v3 = (i3.getEntry(i, j)<l.getEntry(i, j) ? 1 : 0) + (i3.getEntry(i, j)>u.getEntry(i, j) ? 2 : 0);
                int v5 = (i5.getEntry(i, j)<l.getEntry(i, j) ? 1 : 0) + (i5.getEntry(i, j)>u.getEntry(i, j) ? 2 : 0);
                int v7 = (i7.getEntry(i, j)<l.getEntry(i, j) ? 1 : 0) + (i7.getEntry(i, j)>u.getEntry(i, j) ? 2 : 0);
                matrix.setEntry(i, j, v1 + v3*3 + v5*9 + v7*27);
            }
        }
        return matrix;
    }

    private void siltp(BufferedImage image, SuperPixel[] sp) {
        RealMatrix gray = MatrixUtils.createRealMatrix(image.getHeight(), image.getWidth());
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                Color color = new Color(image.getRGB(x, y));
                double v = 0.2989 * color.getRed() + 0.5870 * color.getGreen() + 0.1140 * color.getBlue();
                gray.setEntry(y, x, Math.round(v));
            }
        }

        RealMatrix j3 = siltp(gray, tau, 3*4);
        RealMatrix j5 = siltp(gray, tau, 5*4);

        for (SuperPixel aSp : sp) {
            int[] rows = aSp.rows;
            int[] cols = aSp.cols;

            double[] h3 = new double[total];
            double[] h5 = new double[total];
            for (int k = 0; k < rows.length; k++) {
                h3[(int) j3.getEntry(rows[k], cols[k])]++;
                h5[(int) j5.getEntry(rows[k], cols[k])]++;
            }
            double[] h = Doubles.concat(h3, h5);

            double sum = new Sum().evaluate(h);
            if ((int) sum != 0) {
                for (int k = 0; k < h.length; k++) {
                    h[k] = Math.sqrt(h[k] / sum);
                }
            }

            aSp.features.put(name(), h);
        }
    }

    @Override
    public Type name() {
        return Type.SILTP;
    }

    @Override
    public void extract(BowImage bowImage) {
        siltp(bowImage.image4, bowImage.sp4);
    }
}
