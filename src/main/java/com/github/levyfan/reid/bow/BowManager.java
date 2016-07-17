package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;
import com.github.levyfan.reid.sp.SuperPixelMethond;
import com.google.common.primitives.Doubles;

import java.awt.image.BufferedImage;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * @author fanliwen
 */
public class BowManager {

    private Bow bow;
    private SuperPixelMethond spMethod;
    private StripMethod stripMethod;
    private FeatureManager featureManager;

    public BowManager(
            Bow bow,
            FeatureManager featureManager,
            SuperPixelMethond spMethod,
            int ystep,
            int stripLength) {
        this.bow = bow;
        this.featureManager = featureManager;
        this.spMethod = spMethod;
        this.stripMethod = new StripMethod(ystep, stripLength);
    }

    public Map<Feature.Type, double[]> bow(
            BufferedImage image, BufferedImage mask, Map<Feature.Type, List<double[]>> codebooks) {
        BowImage bowImage = new BowImage(spMethod, stripMethod, image, mask);
        Map<Feature.Type, List<double[]>> features = featureManager.feature(bowImage);

        Map<Feature.Type, double[]> map = new EnumMap<>(Feature.Type.class);
        for (Map.Entry<Feature.Type, List<double[]>> entry : features.entrySet()) {
            List<double[]> feature = bow.bow(bowImage, entry.getValue(), codebooks.get(entry.getKey()));
            map.put(entry.getKey(), Doubles.concat(feature.toArray(new double[0][])));
        }
        return map;
    }

    public double[] fusion(
            BufferedImage image, BufferedImage mask, Map<Feature.Type, List<double[]>> codebooks) {
        Map<Feature.Type, double[]> hists = bow(image, mask, codebooks);

        double[][] features = new double[Feature.Type.values().length][];
        for (Map.Entry<Feature.Type, double[]> entry : hists.entrySet()) {
            features[entry.getKey().ordinal()] = entry.getValue();
        }
        return Doubles.concat(features);
    }
}
