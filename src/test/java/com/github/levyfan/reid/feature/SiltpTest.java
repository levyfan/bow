package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.sp.Slic;
import com.github.levyfan.reid.sp.SuperPixel;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Assert;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;

/**
 * @author fanliwen
 */
public class SiltpTest {

    @Test
    public void testLow() throws IOException, URISyntaxException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/bee.jpg"));
        RealMatrix gray = MatrixUtils.createRealMatrix(image.getHeight(), image.getWidth());
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                Color color = new Color(image.getRGB(x, y));
                double v = 0.2989 * color.getRed() + 0.5870 * color.getGreen() + 0.1140 * color.getBlue();
                gray.setEntry(y, x, Math.round(v));
            }
        }

        RealMatrix j = Siltp.siltp(gray, 0.3, 3);
        Assert.assertTrue(Math.abs(j.getEntry(177-1, 279-1) - 32) < 0.001);
        Assert.assertTrue(Math.abs(j.getEntry(398-1, 170-1) - 0) < 0.001);
    }

    @Test
    public void test() throws IOException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/bee.jpg"));
        SuperPixel[] sp = new Slic(500, 10).slic(image);

        List<double[]> siltp = new Siltp().siltp(image, sp);
        Assert.assertTrue(Math.abs(siltp.get(321-1)[103-1] - 0.0329) < 0.001);
        Assert.assertTrue(Math.abs(siltp.get(288-1)[73-1] - 0.0859) < 0.001);
    }
}
