package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.sp.SuperPixel;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.summary.Sum;
import org.apache.commons.math3.util.Pair;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * @author fanliwen
 */
class Hog implements Feature {

    private static final int LEN = 9;

    private static Pair<RealMatrix, RealMatrix> hog(BufferedImage image) {
        RealMatrix matrix = MatrixUtils.createRealMatrix(image.getHeight(), image.getWidth());
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                Color color = new Color(image.getRGB(x, y));
                double gray = 0.2989 * color.getRed() + 0.5870 * color.getGreen() + 0.1140 * color.getBlue();
                matrix.setEntry(y, x, Math.sqrt(Math.round(gray)));
            }
        }

        RealMatrix iy = MatrixUtils.createRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int i = 0; i < iy.getRowDimension(); i++) {
            for (int j = 1; j < iy.getColumnDimension()-1; j++) {
                double conv = matrix.getEntry(i, j+1) - matrix.getEntry(i, j-1);
                iy.setEntry(i, j, conv);
            }
            iy.setEntry(i, 0, matrix.getEntry(i, 1) - matrix.getEntry(i, 0));
            iy.setEntry(i, iy.getColumnDimension()-1,
                    matrix.getEntry(i, iy.getColumnDimension()-1) - matrix.getEntry(i, iy.getColumnDimension()-2));
        }

        RealMatrix ix = MatrixUtils.createRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int j = 0; j < ix.getColumnDimension(); j++) {
            for (int i = 1; i < ix.getRowDimension()-1; i++) {
                double conv = matrix.getEntry(i+1, j) - matrix.getEntry(i-1, j);
                ix.setEntry(i, j, conv);
            }
            ix.setEntry(0, j, matrix.getEntry(1, j) - matrix.getEntry(0, j));
            ix.setEntry(ix.getRowDimension()-1, j,
                    matrix.getEntry(ix.getRowDimension()-1, j) - matrix.getEntry(ix.getRowDimension()-2, j));
        }

        RealMatrix ied = MatrixUtils.createRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        RealMatrix iphase = MatrixUtils.createRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int i = 0; i < ied.getRowDimension(); i++) {
            for (int j = 0; j < ied.getColumnDimension(); j++) {
                ied.setEntry(i, j, Math.hypot(ix.getEntry(i, j), iy.getEntry(i, j)));
                iphase.setEntry(i, j, Math.atan2(iy.getEntry(i, j), ix.getEntry(i, j)));
            }
        }

        RealMatrix orient = MatrixUtils.createRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int i = 0; i < orient.getRowDimension(); i++) {
            for (int j = 0; j < orient.getColumnDimension(); j++) {
                double percent = iphase.getEntry(i, j) / (2 * Math.PI);
                if (percent < 0) {
                    percent ++;
                }
                double v = Math.ceil(LEN * percent);
                orient.setEntry(i, j, Math.max(Math.min(v, LEN), 1));
            }
        }

        return Pair.create(orient, ied);
    }

    private void hog(BufferedImage image, SuperPixel[] sp) {
        Pair<RealMatrix, RealMatrix> pair = Hog.hog(image);
        RealMatrix orient = pair.getFirst();
        RealMatrix gradient = pair.getSecond();

        for (SuperPixel aSp : sp) {
            double[] tmpBin = new double[LEN];
            for (int k = 0; k < aSp.rows.length; k++) {
                int i = aSp.rows[k];
                int j = aSp.cols[k];
                int index = (int) (orient.getEntry(i, j) - 1);
                tmpBin[index] += gradient.getEntry(i, j);
            }

            double sum = new Sum().evaluate(tmpBin);
            if (sum > 0) {
                for (int k = 0; k < tmpBin.length; k++) {
                    tmpBin[k] = Math.sqrt(tmpBin[k] / sum);
                }
            }
            aSp.features.put(name(), tmpBin);
        }
    }

    @Override
    public Type name() {
        return Type.HOG;
    }

    @Override
    public void extract(BowImage bowImage) {
        hog(bowImage.image4, bowImage.sp4);
    }
}
