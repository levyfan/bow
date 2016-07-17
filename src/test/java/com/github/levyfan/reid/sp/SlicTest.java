package com.github.levyfan.reid.sp;

import org.junit.Assert;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * @author fanliwen
 */
public class SlicTest {

    @Test
    public void test() throws IOException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/bee.jpg"));
        SuperPixel[] sp = new Slic(500, 10).slic(image);

        Assert.assertEquals(488L, sp.length);
    }
}
