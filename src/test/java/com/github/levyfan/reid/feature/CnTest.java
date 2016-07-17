package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.sp.Slic;
import com.github.levyfan.reid.sp.SuperPixel;
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
public class CnTest {

    @Test
    public void test() throws IOException, URISyntaxException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/bee.jpg"));
        SuperPixel[] sp = new Slic(500, 10).slic(image);

        List<double[]> cn = new Cn().cn(image, sp);
        Assert.assertTrue(Math.abs(cn.get(367-1)[3-1] - 0.0594) < 0.001);
        Assert.assertTrue(Math.abs(cn.get(78-1)[11-1] - 0.1575) < 0.001);
    }
}
