package com.github.levyfan.reid.sp;

import com.github.levyfan.reid.feature.Feature;

import java.util.EnumMap;
import java.util.Map;

/**
 * @author fanliwen
 */
public class SuperPixel {
    public int label;
    public int[] rows;
    public int[] cols;

    public Map<Feature.Type, double[]> features = new EnumMap<>(Feature.Type.class);
}
