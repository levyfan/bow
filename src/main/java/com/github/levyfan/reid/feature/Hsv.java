package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.sp.SuperPixel;
import org.apache.commons.math3.stat.descriptive.summary.Sum;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * @author fanliwen
 */
class Hsv implements Feature {

    private void hsv(BufferedImage image, SuperPixel[] sp) {
        for (SuperPixel aSp : sp) {
            double[] tempbin = new double[100];
            int[] rows = aSp.rows;
            int[] columns = aSp.cols;
            for (int k = 0; k < rows.length; k++) {
                Color color = new Color(image.getRGB(columns[k], rows[k]));
                float[] hsbvals = Color.RGBtoHSB(color.getRed(), color.getGreen(), color.getBlue(), null);
                float h = Math.min(hsbvals[0], 0.99f);
                float s = Math.min(hsbvals[1], 0.99f);

                int bin = (int) (Math.floor(h * 10) * 10 + Math.floor(s * 10));
                tempbin[bin]++;
            }

            double sum = new Sum().evaluate(tempbin);
            if ((int) sum != 0) {
                for (int k = 0; k < tempbin.length; k++) {
                    tempbin[k] = Math.sqrt(tempbin[k] / sum);
                }
            }
            aSp.features.put(name(), tempbin);
        }
    }

    @Override
    public String toString() {
        return "Hsv{}";
    }

    @Override
    public Type name() {
        return Type.HSV;
    }

    @Override
    public void extract(BowImage bowImage) {
        hsv(bowImage.image4, bowImage.sp4);
    }
}
