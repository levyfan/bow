package com.github.levyfan.reid.algorithm;

import com.github.levyfan.reid.BowImage;
import com.jmatio.types.MLArray;

import java.util.List;

/**
 * @author fanliwen
 */
public interface Algorithm {

    List<MLArray> train(List<BowImage> bowImages);

    double[] test(BowImage bowImage);
}
