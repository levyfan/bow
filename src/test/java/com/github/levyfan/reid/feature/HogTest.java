package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.sp.Slic;
import com.github.levyfan.reid.sp.SuperPixel;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;
import org.junit.Assert;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;

/**
 * @author fanliwen
 */
public class HogTest {

    @Test
    public void testLow() throws IOException, URISyntaxException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/bee.jpg"));
        Pair<RealMatrix, RealMatrix> pair = Hog.hog(image);
        RealMatrix orient = pair.getFirst();
        RealMatrix gradient = pair.getSecond();

        Assert.assertTrue(Math.abs(orient.getEntry(5-1, 1-1) - 8) < 0.001);
        Assert.assertTrue(Math.abs(gradient.getEntry(5-1, 1-1) - 0.0676) < 0.001);

        Assert.assertTrue(Math.abs(orient.getEntry(9-1, 635-1) - 3) < 0.001);
        Assert.assertTrue(Math.abs(gradient.getEntry(9-1, 635-1) - 0.0467) < 0.001);

        Assert.assertTrue(Math.abs(orient.getEntry(200-1, 300-1) - 7) < 0.001);
        Assert.assertTrue(Math.abs(gradient.getEntry(200-1, 300-1) - 0.1827) < 0.001);
    }

    @Test
    public void test() throws IOException, URISyntaxException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/bee.jpg"));
        SuperPixel[] sp = new Slic(500, 10).slic(image);

        List<double[]> hog = new Hog().hog(image, sp);
        Assert.assertTrue(Math.abs(hog.get(108-1)[1-1] - 0.2638) < 0.001);
        Assert.assertTrue(Math.abs(hog.get(442-1)[5-1] - 0.4386) < 0.001);
    }
}
