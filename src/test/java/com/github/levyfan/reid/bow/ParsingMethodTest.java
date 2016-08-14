package com.github.levyfan.reid.bow;

import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/**
 * @author fanliwen
 */
public class ParsingMethodTest {

    @Test
    public void test() throws IOException {
        BufferedImage mask = ImageIO.read(new File("/data/reid/viper/mask/cam_a/mask_100_0.bmp"));

        Set<Integer> labelsOfSuperpixel = new HashSet<>();
        for (int i =0; i < mask.getHeight(); i++) {
            for (int j = 0; j < mask.getWidth(); j++) {
                Color color = new Color(mask.getRGB(j, i));
                double gray = 0.2989 * color.getRed() + 0.5870 * color.getGreen() + 0.1140 * color.getBlue();
                labelsOfSuperpixel.add((int) gray);
            }
        }

        System.out.println(labelsOfSuperpixel);
    }
}
