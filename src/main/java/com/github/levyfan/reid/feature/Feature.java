package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;

/**
 * @author fanliwen
 */
public interface Feature {

    enum Type {
        HSV, CN, HOG, SILTP, ALL, OPQ_1, OPQ_2, OPQ_3, OPQ_4
    }

    Type name();

    void extract(BowImage bowImage);
}
