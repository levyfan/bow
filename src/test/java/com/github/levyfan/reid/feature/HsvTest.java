package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.sp.Slic;
import com.github.levyfan.reid.sp.SuperPixel;
import org.junit.Assert;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

/**
 * @author fanliwen
 */
public class HsvTest {

    @Test
    public void test() throws IOException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/bee.jpg"));
        SuperPixel[] sp = new Slic(500, 10).slic(image);

        List<double[]> hsv = new Hsv().hsv(image, sp);
        Assert.assertTrue(Math.abs(hsv.get(162-1)[15-1] - 0.0640) < 0.005);
        Assert.assertTrue(Math.abs(hsv.get(219-1)[52-1] - 0.7394) < 0.005);
    }
}
