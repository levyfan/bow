package com.github.levyfan.reid;

import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * @author fanliwen
 */
public class ImageResizeTest {

    @Test
    public void test() throws IOException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/bee.jpg"));
        BufferedImage image4 = ImageUtils.resize(image, 4, false);
        Color color = new Color(image4.getRGB(1000, 500));
        System.out.println(color.getRed() + " " + color.getBlue() + " " + color.getGreen());

        BufferedImage nn4 = ImageUtils.resize(image, 4, true);
        color = new Color(nn4.getRGB(1000, 500));
        System.out.println(color.getRed() + " " + color.getBlue() + " " + color.getGreen());
    }
}
