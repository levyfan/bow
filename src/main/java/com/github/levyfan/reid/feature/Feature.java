package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;

/**
 * @author fanliwen
 */
public interface Feature {

    enum Type {
        HSV, CN, HOG, SILTP, ALL
    }

    Type name();

    void extract(BowImage bowImage);
}
