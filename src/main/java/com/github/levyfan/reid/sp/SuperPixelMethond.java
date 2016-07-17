package com.github.levyfan.reid.sp;

import java.awt.image.BufferedImage;

/**
 * @author fanliwen
 */
public interface SuperPixelMethond {

    SuperPixel[] generate(BufferedImage image);
}
