package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.sp.SuperPixel;

import java.awt.image.BufferedImage;
import java.util.List;

/**
 * @author fanliwen
 */
public interface Feature {

    enum Type {
        HSV, CN, HOG, SILTP
    }

    Type name();

    List<double[]> extract(BufferedImage image, SuperPixel[] sp);

    List<double[]> extract(BowImage bowImage);
}
