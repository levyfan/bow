package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.sp.SuperPixel;
import org.apache.commons.math3.stat.descriptive.summary.Sum;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * @author fanliwen
 */
public class Hsv implements Feature {

    public List<double[]> hsv(BufferedImage image, SuperPixel[] sp) {
        List<double[]> features = new ArrayList<>(sp.length);
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
            features.add(tempbin);
        }
        return features;
    }

    @Override
    public Type name() {
        return Type.HSV;
    }

    @Override
    public List<double[]> extract(BufferedImage image, SuperPixel[] sp) {
        return hsv(image, sp);
    }

    @Override
    public List<double[]> extract(BowImage bowImage) {
        return extract(bowImage.image4, bowImage.sp4);
    }
}
