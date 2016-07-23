package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.sp.SuperPixel;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;

/**
 * @author fanliwen
 */
class Cn implements Feature {

    private RealMatrix w2c;

    Cn() throws IOException, URISyntaxException {
        File file = new File(this.getClass().getResource("/w2c.mat").toURI());
        MatFileReader reader = new MatFileReader(file);
        MLDouble array = (MLDouble) reader.getMLArray("w2c");

        w2c = MatrixUtils.createRealMatrix(array.getM(), array.getN());
        for (int i = 0; i < w2c.getRowDimension(); i++) {
            for (int j = 0; j < w2c.getColumnDimension(); j++) {
                w2c.setEntry(i, j, array.getReal(i, j));
            }
        }
    }

    private void cn(BufferedImage image, SuperPixel[] sp) {
        Table<Integer, Integer, double[]> table = im2c(image, w2c);

        for (SuperPixel aSp : sp) {
            int[] rows = aSp.rows;
            int[] columns = aSp.cols;

            RealMatrix tempCN = MatrixUtils.createRealMatrix(rows.length, w2c.getColumnDimension());
            for (int k = 0; k < rows.length; k++) {
                tempCN.setRow(k, table.get(rows[k], columns[k]));
            }

            double[] tempbin = new double[w2c.getColumnDimension()];
            for (int k = 0; k < tempbin.length; k++) {
                tempbin[k] = Math.sqrt(new Mean().evaluate(tempCN.getColumn(k)));
            }
            aSp.features.put(name(), tempbin);
        }
    }

    private static Table<Integer, Integer, double[]> im2c(BufferedImage image, RealMatrix w2c) {
        Table<Integer, Integer, double[]> table = HashBasedTable.create();
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                Color color = new Color(image.getRGB(x, y));
                int index = (color.getRed()/8) + 32 * (color.getGreen()/8) + 32 * 32 * (color.getBlue()/8);
                table.put(y, x, w2c.getRow(index));
            }
        }
        return table;
    }

    @Override
    public Type name() {
        return Type.CN;
    }

    @Override
    public void extract(BowImage bowImage) {
        cn(bowImage.image4, bowImage.sp4);
    }

    @Override
    public String toString() {
        return "Cn{}";
    }
}
